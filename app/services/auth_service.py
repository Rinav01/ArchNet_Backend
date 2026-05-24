from sqlalchemy.orm import Session
from app.models.user import User
from app.auth.security import hash_password, verify_password, create_access_token

class AuthService:
    @staticmethod
    def signup(db: Session, email: str, username: str, password: str) -> User:
        """Create a new user account if the email and username are unique."""
        # Clean inputs
        email = email.strip().lower()
        username = username.strip()

        # Check existing email
        existing_email = db.query(User).filter(User.email == email).first()
        if existing_email:
            raise ValueError("An account with this email already exists.")

        # Check existing username
        existing_username = db.query(User).filter(User.username == username).first()
        if existing_username:
            raise ValueError("Username is already taken.")

        # Create new user
        password_hash = hash_password(password)
        new_user = User(
            email=email,
            username=username,
            password_hash=password_hash,
            preferences={}
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user

    @staticmethod
    def login(db: Session, email: str, password: str) -> tuple[str, User]:
        """Authenticate a user by email and password, returning a JWT token and user instance."""
        email = email.strip().lower()
        
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise ValueError("Invalid email or password.")
            
        if not verify_password(password, user.password_hash):
            raise ValueError("Invalid email or password.")
            
        token = create_access_token(user_id=str(user.id), email=user.email)
        return token, user
