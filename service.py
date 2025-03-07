import os

from notion_client import Client

from notion.schema import NotionPage

client = Client(auth=os.getenv("NOTION_INTEGRATION_SECRET"))

TASK_DATABASE_ID = "938ee40926734a9d8f29984755964a0d"
BD_DATABASE_ID = "84e09969bdb9482a84b0c97195e306a9"
PROJECT_DATABASE_ID = "9039d2cfa72a4e8ebc51ce4c632f1fc3"

STATUS_PROJECT_NOT_DONE = ["Planning", "In progress", "Paused"]
STATUS_BD_NOT_DONE = ["In talks", "In progress"]
STATUS_TASK_NOT_DONE = [
    "Not started",
    "In progress",
    "Planned",
    "In review",
    "Approved",
]


def query(db, where=None):
    """Query Notion database, return NotionPage objects"""
    kwargs = {}
    if where:
        kwargs["filter"] = where

    payload = client.databases.query(database_id=db, **kwargs)
    pages = [NotionPage(**result) for result in payload["results"]]
    return pages


def update(pages: list):
    for page in pages:
        if page.changed:
            print(page.updated_fields)
            client.pages.update(page_id=page.id, properties=page.updated_fields)


# filters
def filter_only_features_in_spaces(space_ids: list) -> dict:
    """Build a filter that only queries task with specific Space ID"""
    return {
        "and": [
            {"property": "Feature-type", "relation": {"is_not_empty": True}},
            {
                "or": [
                    {"property": "Space ID", "formula": {"number": {"equals": x}}}
                    for x in space_ids
                ]
            },
        ],
    }


def filter_only_tasks() -> dict:
    """Build a Notion filter that only queries tasks"""
    return {"property": "Feature-type", "relation": {"is_empty": True}}


def filter_by_status(statuses: list) -> dict:
    """Build a Notion filter that only queries table with specific status"""
    return {"or": [{"property": "Status", "status": {"equals": x}} for x in statuses]}


# legacy schema
def update_sync_status(page):
    client.pages.update(page_id=page["id"], properties={"Synced": {"checkbox": True}})


def update_feature_insights(page):
    client.pages.update(
        page_id=page["id"],
        properties={
            "Hours Spent": {"number": page["hours"]},
            "Report Periods": {"number": page["periods"]},
            "Tasks Closed": {"number": page["closed"]},
            "Tasks Open": {"number": page["open"]},
        },
    )
