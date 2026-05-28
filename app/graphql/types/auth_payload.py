import strawberry
from app.graphql.types.user_type import UserType

@strawberry.type
class AuthPayload:
    token: str
    refresh_token: str | None = None
    user: UserType
