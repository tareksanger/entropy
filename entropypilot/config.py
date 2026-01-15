from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

environment_dir = Path(__file__).parent

class Config(BaseSettings):
  
    openai_api_key: str

    model_config = SettingsConfigDict(
        env_file=(f"{environment_dir}/.env", f"{environment_dir}/.env.local"),
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
config = Config() # type: ignore