import strawberry
from app.graphql.types.user_type import UserType

@strawberry.type
class AuthPayload:
    token: str
    user: UserType
