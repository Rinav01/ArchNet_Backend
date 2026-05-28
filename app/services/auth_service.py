import uuid
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.user import User
from app.auth.security import hash_password, verify_password, create_access_token

class AuthService:
    @staticmethod
    def signup(db: Session, email: str, username: str, password: str) -> User:
        """Create a new user account if the email and username are unique."""
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

    @classmethod
    def generate_tokens(cls, db: Session, user: User) -> tuple[str, str]:
        """Generates a short-lived access token and a secure long-lived refresh token,
        persisting the rotated refresh token on the user's database record.
        """
        access_token = create_access_token(user_id=str(user.id), email=user.email)
        
        # Secure unique hex string for rotated token
        refresh_token = uuid.uuid4().hex + uuid.uuid4().hex
        expiry = datetime.utcnow() + timedelta(days=7)
        
        user.refresh_token = refresh_token
        user.refresh_token_expires_at = expiry
        db.commit()
        
        return access_token, refresh_token

    @classmethod
    def login(cls, db: Session, email: str, password: str) -> tuple[str, str, User]:
        """Authenticate a user by email and password, returning an access token, refresh token, and user instance."""
        email = email.strip().lower()
        
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise ValueError("Invalid email or password.")
            
        if not verify_password(password, user.password_hash):
            raise ValueError("Invalid email or password.")
            
        access_token, refresh_token = cls.generate_tokens(db, user)
        return access_token, refresh_token, user

    @classmethod
    def rotate_refresh_token(cls, db: Session, refresh_token: str) -> tuple[str, str, User]:
        """Performs single-use refresh token rotation, invalidating the old token immediately
        and generating a fresh pair of access and refresh tokens.
        """
        user = db.query(User).filter(User.refresh_token == refresh_token).first()
        if not user:
            raise ValueError("Invalid or revoked refresh token.")
            
        if user.refresh_token_expires_at and user.refresh_token_expires_at < datetime.utcnow():
            user.refresh_token = None
            user.refresh_token_expires_at = None
            db.commit()
            raise ValueError("Refresh token has expired.")
            
        # Single-use revocation & fresh tokens issuance
        access_token, new_refresh_token = cls.generate_tokens(db, user)
        return access_token, new_refresh_token, user
