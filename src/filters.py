from pydantic import BaseModel, model_validator
from qdrant.http import models as qmodels

class MetadataFilter(BaseModel):
    filename: str | None = None
    filenames: list[str] | None = None
    page: int | None = None
    section: str | None = None
    document_id: str | None = None

    @model_validator(mode=after)
    def normalize(self):
        names=[n.strip() for n in (self.filenames or []) if isinstance(n,str) and n.strip()]

        if not names:
            self.filenames is None
        elif len(names)==1:
            self.filename=names[0]
            self.filenames=None

        else:
            self.filename=None
            self.filenames=names
            self.page=None

        if self.filename is not None:
            self.filename=(self.filename.strip() or None)

        if self.document_id is not None:
            self.document_id = (
                self.document_id.strip() or None
            )

        return self

def filters_to_dict(filters):
    if filters is None:
        return None
    flat=filters.model_dump(exclude=None)

    return flat or None

def filters_to_qdrant(filters):
    flat=filters_to_dict(filters)

    if not flat:
        return None
    conditions=[]
    for fiel,value in flat.items():
        if (
            field == "filenames"
            and isinstance(value, list)
        ):
            conditions.append(qmodels.FieldCondition(
                key="metadata.filename",
                match=qmodels.MatchAny(
                    any=value
                )
            ))
        elif isinstance(value,(str,int)):
            conditions.append(qmodels.FieldCondition(
                key=f"metadata.{field}",
                match=qmodels.MatchValue(
                    value=value
                )
            ))
            
    if not conditions:
        return None

    return qmodels.Filter(
        must=conditions
    )
        



