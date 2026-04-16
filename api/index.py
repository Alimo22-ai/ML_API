import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from meal_model import MealRecommendationSystem, UserNutritionModel, UserProfile
from chatbot_engine import FoodChatbot


APP_TITLE = "Healthy Food AI API"
APP_VERSION = "1.0.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "nutrition_model.joblib")
RECIPES_PATH = os.path.join(PROJECT_ROOT, "recips.json")


class UserProfileRequest(BaseModel):
    age: int = Field(..., ge=1, le=120)
    gender: str
    height: float = Field(..., gt=0)
    weight: float = Field(..., gt=0)
    activity_level: str
    fitness_goal: str
    dietary_preference: str
    allergies: List[str] = Field(default_factory=list)
    health_conditions: List[str] = Field(default_factory=list)
    meals_per_day: int = Field(default=3, ge=1, le=6)
    notes: str = ""
    max_calories: Optional[float] = Field(default=None, gt=0)
    max_protein: Optional[float] = Field(default=None, gt=0)
    max_carbs: Optional[float] = Field(default=None, gt=0)
    max_fats: Optional[float] = Field(default=None, gt=0)

    def to_user_profile(self) -> UserProfile:
        return UserProfile(
            age=self.age,
            gender=self.gender,
            height=self.height,
            weight=self.weight,
            activity_level=self.activity_level,
            fitness_goal=self.fitness_goal,
            dietary_preference=self.dietary_preference,
            allergies=self.allergies,
            health_conditions=self.health_conditions,
            meals_per_day=self.meals_per_day,
            notes=self.notes,
            max_calories=self.max_calories,
            max_protein=self.max_protein,
            max_carbs=self.max_carbs,
            max_fats=self.max_fats,
        )


class ChatRequest(BaseModel):
    message: str
    user_profile: Optional[UserProfileRequest] = None


def load_recipes(recipes_path: str) -> List[Dict[str, Any]]:
    import json

    if not os.path.exists(recipes_path):
        raise FileNotFoundError(
            f"Recipes file not found: {recipes_path}. "
            "Make sure recips.json exists in the project root."
        )

    with open(recipes_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("recips.json must contain a list of recipes.")
    return data


def create_app() -> FastAPI:
    app = FastAPI(title=APP_TITLE, version=APP_VERSION)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: Dict[str, Any] = {
        "model": None,
        "recommender_system": None,
        "chatbot": None,
        "recipes": [],
        "startup_error": None,
    }

    @app.on_event("startup")
    def startup_event() -> None:
        state["startup_error"] = None
        state["recipes"] = []
        state["chatbot"] = None
        state["recommender_system"] = None
        state["model"] = None

        try:
            recipes = load_recipes(RECIPES_PATH)
            state["recipes"] = recipes
            state["chatbot"] = FoodChatbot(foods_data=recipes)

            if os.path.exists(MODEL_PATH):
                try:
                    state["recommender_system"] = MealRecommendationSystem(
                        model_path=MODEL_PATH,
                        recipes_path=RECIPES_PATH,
                    )
                except Exception as exc:
                    state["recommender_system"] = None
                    state["startup_error"] = f"Failed to initialize recommendation system: {exc}"

                try:
                    model = UserNutritionModel()
                    model.load(MODEL_PATH)
                    state["model"] = model
                except Exception as exc:
                    state["model"] = None
                    if state["startup_error"]:
                        state["startup_error"] += f" | Failed to load model: {exc}"
                    else:
                        state["startup_error"] = f"Failed to load model: {exc}"
            else:
                state["startup_error"] = f"Model file not found: {MODEL_PATH}"

        except Exception as exc:
            state["recipes"] = []
            state["chatbot"] = None
            state["recommender_system"] = None
            state["model"] = None
            state["startup_error"] = f"Startup error: {exc}"

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "message": "Healthy Food AI API is running",
            "version": APP_VERSION,
            "model_loaded": state["model"] is not None,
            "recipes_loaded": len(state["recipes"]),
            "startup_error": state.get("startup_error"),
        }

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": state["model"] is not None,
            "recipes_loaded": len(state["recipes"]),
            "model_path": MODEL_PATH,
            "recipes_path": RECIPES_PATH,
            "model_exists": os.path.exists(MODEL_PATH),
            "recipes_exists": os.path.exists(RECIPES_PATH),
            "startup_error": state.get("startup_error"),
        }

    @app.post("/predict-targets")
    def predict_targets(payload: UserProfileRequest) -> Dict[str, Any]:
        if state["model"] is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Model is not loaded.",
                    "startup_error": state.get("startup_error"),
                    "model_path": MODEL_PATH,
                    "model_exists": os.path.exists(MODEL_PATH),
                },
            )

        try:
            user = payload.to_user_profile()
            daily_targets = state["model"].predict_daily_targets(user)
            meals = max(user.meals_per_day, 1)

            per_meal_target = {
                "calories": round(daily_targets["calories"] / meals, 1),
                "protein": round(daily_targets["protein"] / meals, 1),
                "carbs": round(daily_targets["carbs"] / meals, 1),
                "fats": round(daily_targets["fats"] / meals, 1),
            }

            return {
                "daily_targets": {k: round(v, 1) for k, v in daily_targets.items()},
                "per_meal_target": per_meal_target,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/recommend")
    def recommend(payload: UserProfileRequest) -> Dict[str, Any]:
        if state["recommender_system"] is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Recommendation system is not loaded.",
                    "startup_error": state.get("startup_error"),
                    "model_path": MODEL_PATH,
                    "recipes_path": RECIPES_PATH,
                    "model_exists": os.path.exists(MODEL_PATH),
                    "recipes_exists": os.path.exists(RECIPES_PATH),
                },
            )

        try:
            user = payload.to_user_profile()
            result = state["recommender_system"].recommend(user, top_k=5)
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/chat")
    def chat(payload: ChatRequest) -> Dict[str, Any]:
        if state["chatbot"] is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Chatbot is not initialized.",
                    "startup_error": state.get("startup_error"),
                    "recipes_path": RECIPES_PATH,
                    "recipes_exists": os.path.exists(RECIPES_PATH),
                },
            )

        try:
            user_profile = payload.user_profile.to_user_profile() if payload.user_profile else None
            answer = state["chatbot"].respond(
                user_msg=payload.message,
                user_profile=user_profile.__dict__ if user_profile else None,
            )
            return {"reply": answer}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/recipes")
    def recipes(limit: int = 10) -> Dict[str, Any]:
        limit = max(1, min(limit, 100))
        return {
            "count": len(state["recipes"]),
            "items": state["recipes"][:limit],
            "startup_error": state.get("startup_error"),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
