from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    debug: bool = False
    allowed_origins: str  # comma-separated list of origins allowed to call this API

    class Config:
        env_file = ".env"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


settings = Settings()
