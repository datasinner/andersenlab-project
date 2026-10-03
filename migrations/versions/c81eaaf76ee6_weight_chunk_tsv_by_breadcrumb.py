"""weight chunk tsv by breadcrumb

Revision ID: c81eaaf76ee6
Revises: 52e2b6e8b9a9
Create Date: 2026-10-02 12:31:07.980854

"""
from typing import Sequence, Union

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'c81eaaf76ee6'
down_revision: Union[str, Sequence[str], None] = '52e2b6e8b9a9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


NEW_EXPRESSION = (
    "setweight(to_tsvector('english', translate(split_part(content, E'\\n\\n', 1), '/', ' ')), 'A')"
    " || setweight(to_tsvector('english', translate(content, '/', ' ')), 'B')"
)
OLD_EXPRESSION = "to_tsvector('english', content)"


def _replace_tsv(expression: str) -> None:
    # A generated column's expression can't be altered in place (before
    # PostgreSQL 17), so drop and re-add it; existing rows are recomputed.
    op.drop_index("ix_chunk_tsv", table_name="chunk", postgresql_using="gin")
    op.drop_column("chunk", "tsv")
    op.add_column(
        "chunk",
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed(expression, persisted=True),
            nullable=False,
        ),
    )
    op.create_index("ix_chunk_tsv", "chunk", ["tsv"], unique=False, postgresql_using="gin")


def upgrade() -> None:
    """Weight the breadcrumb above the body and split words joined by "/"."""
    _replace_tsv(NEW_EXPRESSION)


def downgrade() -> None:
    """Downgrade schema."""
    _replace_tsv(OLD_EXPRESSION)
