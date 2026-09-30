"""
Search over the data catalog, the way Purview or Unity Catalog let you type
"rain" and find every table, column and metric about it.

The index is built from what is published in the warehouse (DuckDB
COMMENTs, gold.metric_definitions, gold.lineage_edges) plus the glossary and
lineage labels in semantic_layer.yml - so search results can never disagree
with what the pipeline documented and checked.

Ranking is plain keyword scoring, no LLM or embeddings: every search word
(or one of its glossary synonyms) must match somewhere, and a match in an
asset's name counts more than one in its description or columns.
"""

from __future__ import annotations

import re

import duckdb

SCHEMAS = ("bronze", "silver", "gold", "ops")
STOPWORDS = {"a", "an", "and", "the", "of", "in", "on", "for", "to", "is", "by", "with", "what", "which", "where", "data"}

# How much a match counts, by where it was found.
NAME_WEIGHT, DESCRIPTION_WEIGHT, COLUMN_NAME_WEIGHT, COLUMN_DESCRIPTION_WEIGHT = 10, 4, 3, 1.5
SYNONYM_DISCOUNT = 0.6  # a synonym hit counts a bit less than the word actually typed
COLUMN_BONUS, MAX_COLUMN_BONUS = 0.5, 2  # a table with several matching columns is about the topic

NON_TABLE_KINDS = {"source": "Source", "service": "Service", "consumer": "Consumer", "file": "File"}


def build_index(con: duckdb.DuckDBPyConnection, layer: dict) -> dict:
    """All catalog assets keyed by id: tables and views in every layer,
    metrics, and the non-table lineage nodes (APIs, apps, files)."""
    lineage = layer["lineage"]
    edges = con.execute("SELECT upstream, downstream FROM gold.lineage_edges").fetchall()
    upstream_of, downstream_of = {}, {}
    for up, down in edges:
        upstream_of.setdefault(down, []).append(up)
        downstream_of.setdefault(up, []).append(down)

    placeholders = ", ".join("?" * len(SCHEMAS))
    objects = con.execute(
        f"""
        SELECT schema_name, table_name, 'Table', comment FROM duckdb_tables()
        WHERE database_name = current_database() AND schema_name IN ({placeholders})
        UNION ALL
        SELECT schema_name, view_name, 'View', comment FROM duckdb_views()
        WHERE database_name = current_database() AND schema_name IN ({placeholders}) AND NOT internal
        """,
        [*SCHEMAS, *SCHEMAS],
    ).fetchall()
    columns = {}
    for schema, table, column, data_type, comment in con.execute(
        f"""
        SELECT schema_name, table_name, column_name, data_type, comment FROM duckdb_columns()
        WHERE database_name = current_database() AND schema_name IN ({placeholders}) ORDER BY column_index
        """,
        list(SCHEMAS),
    ).fetchall():
        columns.setdefault(f"{schema}.{table}", []).append({"name": column, "type": data_type, "description": comment or ""})

    assets = {}
    for schema, name, kind, comment in objects:
        asset_id = f"{schema}.{name}"
        assets[asset_id] = {
            "id": asset_id, "name": asset_id, "kind": kind, "layer": schema, "description": comment or "",
            "columns": columns.get(asset_id, []), "built_by": lineage.get(asset_id, {}).get("built_by"),
            "upstream": upstream_of.get(asset_id, []), "downstream": downstream_of.get(asset_id, []),
        }

    for name, label, description, table, expression, filter_sql, unit in con.execute(
        "SELECT name, label, description, table_name, expression, filter, unit FROM gold.metric_definitions"
    ).fetchall():
        asset_id = f"metric:{name}"
        assets[asset_id] = {
            "id": asset_id, "name": label, "kind": "Metric", "layer": "semantic layer", "description": description,
            "columns": [], "metric": {"name": name, "table": table, "expression": expression, "filter": filter_sql, "unit": unit},
            "upstream": [table], "downstream": [],
        }
        if table in assets:
            assets[table].setdefault("metrics", []).append(asset_id)

    for node, spec in lineage.items():
        if spec["type"] in NON_TABLE_KINDS and node not in assets:
            assets[node] = {
                "id": node, "name": spec.get("label", node), "kind": NON_TABLE_KINDS[spec["type"]], "layer": "outside the warehouse",
                "description": spec.get("description", ""), "columns": [], "built_by": spec.get("built_by"),
                "upstream": upstream_of.get(node, []), "downstream": downstream_of.get(node, []),
            }
    return assets


def search_terms(query: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", query.lower()) if t not in STOPWORDS]


def expand(term: str, glossary: dict) -> dict[str, float]:
    """The term plus every word in any glossary group it belongs to, each with its weight."""
    words = {term: 1.0}
    for key, synonyms in glossary.items():
        group = [key, *synonyms]
        if term in group:
            words.update({w: SYNONYM_DISCOUNT for w in group if w != term})
    return words


def _hits(word: str, text: str) -> bool:
    """Word-start match, with _ as a separator: "precip" finds precipitation_sum_mm,
    but "rain" does not find training_rows."""
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}", text.replace("_", " ")) is not None


def score(asset: dict, terms: list[str], glossary: dict) -> tuple[float, list[str]]:
    """(relevance, matching column names). 0 if any search word matches nothing."""
    name = f"{asset['name']} {asset['id']}".lower()
    description = f"{asset['description']} " + " ".join(str(v) for v in asset.get("metric", {}).values())
    description = description.lower()
    total, matched_columns = 0.0, []
    for term in terms:
        best = 0.0
        for word, weight in expand(term, glossary).items():
            if _hits(word, name):
                best = max(best, NAME_WEIGHT * weight)
            if _hits(word, description):
                best = max(best, DESCRIPTION_WEIGHT * weight)
            for column in asset["columns"]:
                if _hits(word, column["name"].lower()):
                    best = max(best, COLUMN_NAME_WEIGHT * weight)
                elif _hits(word, column["description"].lower()):
                    best = max(best, COLUMN_DESCRIPTION_WEIGHT * weight)
                else:
                    continue
                if column["name"] not in matched_columns:
                    matched_columns.append(column["name"])
        if best == 0:
            return 0.0, []
        total += best
    return total + min(COLUMN_BONUS * len(matched_columns), MAX_COLUMN_BONUS), matched_columns


def search(assets: dict, query: str, glossary: dict) -> list[tuple[dict, float, list[str]]]:
    """Matching assets, best first. An empty query returns everything, by layer."""
    terms = search_terms(query)
    if not terms:
        order = {layer: i for i, layer in enumerate(("gold", "semantic layer", "silver", "bronze", "ops"))}
        return [(a, 0.0, []) for a in sorted(assets.values(), key=lambda a: (order.get(a["layer"], 9), a["name"]))]
    results = []
    for asset in assets.values():
        relevance, matched_columns = score(asset, terms, glossary)
        if relevance:
            results.append((asset, relevance, matched_columns))
    return sorted(results, key=lambda r: (-r[1], r[0]["name"]))


def matched_words(query: str, glossary: dict) -> list[str]:
    """Every word a query effectively searched for, for highlighting."""
    return sorted({w for term in search_terms(query) for w in expand(term, glossary)}, key=len, reverse=True)
