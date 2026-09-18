from recommendation.app.server import Lab


def test_presentation_labels_are_bounded_and_human_readable():
    annotation = {
        "concepts": ["concept:fallback-topic~hash"],
        "format": "format:news-analysis~hash",
        "provenance": {
            "canonicalization": {
                "mappings": {
                    "concepts": [
                        {"lexical": "climate policy"},
                        {"lexical": "energy transition"},
                        {"lexical": "carbon markets"},
                        {"lexical": "global summit"},
                        {"lexical": "public finance"},
                        {"lexical": "not returned"},
                    ],
                    "audiences": [{"lexical": "policy readers"}],
                }
            }
        },
    }

    result = Lab._presentation_from_annotation(annotation)

    assert result["concepts"] == [
        "climate policy",
        "energy transition",
        "carbon markets",
        "global summit",
        "public finance",
    ]
    assert result["audiences"] == ["policy readers"]
    assert result["semantic_format"] == "news analysis"


def test_presentation_falls_back_to_portable_ids():
    result = Lab._presentation_from_annotation({
        "concepts": ["concept:world-series~digest"],
        "event_types": ["event:sports-team-recovery~digest"],
    })

    assert result["concepts"] == ["world series"]
    assert result["events"] == ["sports team recovery"]

