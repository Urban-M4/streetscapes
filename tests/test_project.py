import pandas as pd
from streetscapes.project import Project


def test_project_ingest_and_filter_bbox(tmp_path):
    db_path = tmp_path / "test.duckdb"
    project = Project(db_path=str(db_path))

    # Create fake data
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "geometry": "POINT(4.89 52.37)",
                "computed_geometry": "POINT(4.89 52.37)",
            },
            {
                "id": "2",
                "geometry": "POINT(4.91 52.39)",
                "computed_geometry": "POINT(4.91 52.39)",
            },
        ]
    )

    project.ingest_mapillary(df)

    # Filter within bbox
    bbox = (4.88, 52.36, 4.90, 52.38)
    filtered = project.filter_bbox("mapillary", bbox)

    count = filtered.count().execute()
    assert count == 1
