
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None




TARGET_COLUMNS = ["Daily Calorie Target", "Protein", "Carbohydrates", "Fat"]
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Height",
    "Weight",
    "Activity Level",
    "Fitness Goal",
    "Dietary Preference",
]


@dataclass
class UserProfile:
    age: int
    gender: str
    height: float
    weight: float
    activity_level: str
    fitness_goal: str
    dietary_preference: str
    allergies: List[str]
    health_conditions: List[str]
    meals_per_day: int = 3
    notes: str = ""
    max_calories: Optional[float] = None
    max_protein: Optional[float] = None
    max_carbs: Optional[float] = None
    max_fats: Optional[float] = None

    def normalized_allergies(self) -> List[str]:
        return [a.strip().lower() for a in self.allergies if str(a).strip()]

    def normalized_conditions(self) -> List[str]:
        return [c.strip().lower() for c in self.health_conditions if str(c).strip()]


def normalize_recipe_lists(recipe: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(recipe)
    for key in ["diet", "allergy", "diseases", "categories", "ingredients"]:
        value = normalized.get(key, [])
        if isinstance(value, list):
            normalized[key] = [str(v).strip().lower() for v in value]
        else:
            normalized[key] = []
    for key in ["name", "image"]:
        normalized[key] = str(normalized.get(key, ""))
    return normalized


def _build_base_preprocessor() -> ColumnTransformer:
    numeric_features = ["Age", "Height", "Weight"]
    categorical_features = [
        "Gender",
        "Activity Level",
        "Fitness Goal",
        "Dietary Preference",
    ]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )


def build_regressor(model_name: str):
    model_name = model_name.strip().lower()

    if model_name == "random_forest":
        return MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=250,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            ) from exc

        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )
        )




class UserNutritionModel:
    def __init__(self, model_name: str = "random_forest") -> None:
        self.model_name = model_name
        self.pipeline: Optional[Pipeline] = None

    @staticmethod
    def load_training_dataframe(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]

        missing = [c for c in FEATURE_COLUMNS + TARGET_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in nutrition dataset: {missing}")

        # Remove repeated header rows and coerce numeric columns
        df = df[df["Age"].astype(str).str.lower() != "age"].copy()
        for col in ["Age", "Height", "Weight"] + TARGET_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Age", "Height", "Weight"] + TARGET_COLUMNS)
        return df.reset_index(drop=True)

    def train(self, csv_path: str) -> None:
        df = self.load_training_dataframe(csv_path)
        X = df[FEATURE_COLUMNS].copy()
        y = df[TARGET_COLUMNS].copy()

        self.pipeline = Pipeline([
            ("preprocessor", _build_base_preprocessor()),
            ("model", build_regressor(self.model_name)),
        ])
        self.pipeline.fit(X, y)

    def evaluate(self, csv_path: str, cv: int = 5) -> Dict[str, Any]:
        """Run *cv*-fold cross-validation and return per-target metrics."""
        df = self.load_training_dataframe(csv_path)
        X = df[FEATURE_COLUMNS].copy()
        y = df[TARGET_COLUMNS].copy()

        pipeline = Pipeline([
            ("preprocessor", _build_base_preprocessor()),
            ("model", build_regressor(self.model_name)),
        ])

        y_pred = cross_val_predict(pipeline, X, y, cv=cv)

        target_names = ["calories", "protein", "carbs", "fats"]
        metrics: Dict[str, Any] = {"model_name": self.model_name, "cv_folds": cv, "targets": {}}
        overall_r2 = []

        for idx, name in enumerate(target_names):
            y_true_col = y.iloc[:, idx].values
            y_pred_col = y_pred[:, idx]
            mae = float(mean_absolute_error(y_true_col, y_pred_col))
            rmse = float(np.sqrt(mean_squared_error(y_true_col, y_pred_col)))
            r2 = float(r2_score(y_true_col, y_pred_col))
            overall_r2.append(r2)
            metrics["targets"][name] = {
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R2": round(r2, 4),
            }

        metrics["average_r2"] = round(float(np.mean(overall_r2)), 4)
        return metrics

    def fit_dataframe(self, df: pd.DataFrame) -> None:
        X = df[FEATURE_COLUMNS].copy()
        y = df[TARGET_COLUMNS].copy()

        self.pipeline = Pipeline([
            ("preprocessor", _build_base_preprocessor()),
            ("model", build_regressor(self.model_name)),
        ])
        self.pipeline.fit(X, y)

    def predict_daily_targets(self, user: UserProfile) -> Dict[str, float]:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained or loaded.")

        X_new = pd.DataFrame([
            {
                "Age": user.age,
                "Gender": user.gender,
                "Height": user.height,
                "Weight": user.weight,
                "Activity Level": user.activity_level,
                "Fitness Goal": user.fitness_goal,
                "Dietary Preference": user.dietary_preference,
            }
        ])

        pred = self.pipeline.predict(X_new)[0]
        daily = {
            "calories": max(float(pred[0]), 900.0),
            "protein": max(float(pred[1]), 20.0),
            "carbs": max(float(pred[2]), 30.0),
            "fats": max(float(pred[3]), 10.0),
        }
        return daily

    def save(self, path: str) -> None:
        if self.pipeline is None:
            raise RuntimeError("No trained model to save.")
        payload = {
            "model_name": self.model_name,
            "pipeline": self.pipeline,
        }
        joblib.dump(payload, path)

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        if isinstance(payload, dict) and "pipeline" in payload:
            self.model_name = payload.get("model_name", "random_forest")
            self.pipeline = payload["pipeline"]
        else:
            # Backward compatibility with old saved files
            self.pipeline = payload
            self.model_name = "random_forest"


class RecipeRecommender:
    def __init__(self, recipes_path: str) -> None:
        with open(recipes_path, "r", encoding="utf-8") as f:
            self.recipes = [normalize_recipe_lists(r) for r in json.load(f)]

    @staticmethod
    def _contains_any(text: str, keywords: List[str]) -> bool:
        text = text.lower()
        return any(k in text for k in keywords)

    def _allergy_conflict(self, recipe: Dict[str, Any], user: UserProfile) -> bool:
        recipe_allergy = recipe.get("allergy", [])
        ingredients_text = " ".join(recipe.get("ingredients", []))
        allergy_map = {
            "nuts": ["nuts", "nut", "almond", "walnut", "cashew", "peanut", "pistachio", "hazelnut"],
            "lactose": ["lactose", "milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "feta", "parmesan"],
            "lactuse": ["lactose", "milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "feta", "parmesan"],
            "gluten": ["gluten", "wheat", "barley", "rye", "bread", "pasta", "flour", "bun", "granola"],
            "eggs": ["egg", "eggs", "omelette", "mayo", "mayonnaise"],
        }
        for allergy in user.normalized_allergies():
            if allergy == "none":
                continue
            if allergy in recipe_allergy:
                return True
            keywords = allergy_map.get(allergy, [allergy])
            if self._contains_any(ingredients_text, keywords):
                return True
        return False

    def _diet_match(self, recipe: Dict[str, Any], user: UserProfile) -> bool:
        pref = user.dietary_preference.strip().lower()
        recipe_diets = recipe.get("diet", [])
        ingredients_text = " ".join(recipe.get("ingredients", []))

        if pref in {"", "none"}:
            return True
        if pref == "vegan":
            return "vegan" in recipe_diets
        if pref == "keto":
            return "keto" in recipe_diets or recipe.get("carbs", 999) <= 20
        if pref == "low carb":
            return recipe.get("carbs", 999) <= 35
        if pref == "high protein":
            return recipe.get("protein", 0) >= 18
        return True

    def _health_safe(self, recipe: Dict[str, Any], user: UserProfile) -> bool:
        conditions = set(user.normalized_conditions())
        carbs = float(recipe.get("carbs", 0) or 0)
        fats = float(recipe.get("fats", 0) or 0)

        if "diabetes" in conditions and carbs > 45:
            return False
        if ("heart" in conditions or "heart disease" in conditions) and fats > 30:
            return False
        if "hypertension" in conditions and fats > 25:
            return False
        return True

    @staticmethod
    def _relative_match(actual: float, target: float) -> float:
        if target <= 0:
            return 0.0
        return max(0.0, 1.0 - abs(actual - target) / max(target, 1.0))

    def _score_recipe(
        self,
        recipe: Dict[str, Any],
        per_meal_target: Dict[str, float],
        user: UserProfile,
    ) -> float:
        cal_score = self._relative_match(float(recipe.get("calories", 0)), per_meal_target["calories"])
        protein_score = self._relative_match(float(recipe.get("protein", 0)), per_meal_target["protein"])
        carbs_score = self._relative_match(float(recipe.get("carbs", 0)), per_meal_target["carbs"])
        fats_score = self._relative_match(float(recipe.get("fats", 0)), per_meal_target["fats"])

        bonus = 0.0
        if self._diet_match(recipe, user):
            bonus += 0.08
        if self._health_safe(recipe, user):
            bonus += 0.05
        if recipe.get("discount"):
            bonus += 0.02

        goal = user.fitness_goal.lower()
        if "lose" in goal or "loss" in goal:
            if float(recipe.get("calories", 0)) <= per_meal_target["calories"]:
                bonus += 0.05
        elif "muscle" in goal or "gain" in goal:
            if float(recipe.get("protein", 0)) >= per_meal_target["protein"] * 0.8:
                bonus += 0.05

        return 0.35 * cal_score + 0.30 * protein_score + 0.20 * carbs_score + 0.15 * fats_score + bonus

    def _build_reason(self, recipe: Dict[str, Any], per_meal_target: Dict[str, float], user: UserProfile) -> str:
        reasons: List[str] = []
        protein = float(recipe.get("protein", 0))
        carbs = float(recipe.get("carbs", 0))
        calories = float(recipe.get("calories", 0))

        if protein >= per_meal_target["protein"] * 0.8:
            reasons.append("good protein match")
        if calories <= per_meal_target["calories"] * 1.1:
            reasons.append("fits meal calories")
        if user.dietary_preference.strip().lower() in recipe.get("diet", []):
            reasons.append(f"matches {user.dietary_preference} preference")
        if "diabetes" in user.normalized_conditions() and carbs <= 35:
            reasons.append("better carb level for diabetes")
        if ("heart" in user.normalized_conditions() or "heart disease" in user.normalized_conditions()) and float(recipe.get("fats", 0)) <= 20:
            reasons.append("lighter fat profile")
        if not reasons:
            reasons.append("closest macro match from the recipe dataset")
        return ", ".join(reasons[:3])

    def recommend(self, user: UserProfile, daily_targets: Dict[str, float], top_k: int = 5) -> Dict[str, Any]:
        meals = max(int(user.meals_per_day), 1)
        per_meal_target = {
            "calories": daily_targets["calories"] / meals,
            "protein": daily_targets["protein"] / meals,
            "carbs": daily_targets["carbs"] / meals,
            "fats": daily_targets["fats"] / meals,
        }

        candidates = []
        for recipe in self.recipes:
            if self._allergy_conflict(recipe, user):
                continue
            if not self._diet_match(recipe, user):
                continue
            if not self._health_safe(recipe, user):
                continue

            if user.max_calories is not None and float(recipe.get("calories", 0)) > user.max_calories:
                continue
            if user.max_protein is not None and float(recipe.get("protein", 0)) > user.max_protein:
                continue
            if user.max_carbs is not None and float(recipe.get("carbs", 0)) > user.max_carbs:
                continue
            if user.max_fats is not None and float(recipe.get("fats", 0)) > user.max_fats:
                continue

            score = self._score_recipe(recipe, per_meal_target, user)
            item = dict(recipe)
            item["score"] = round(float(score), 4)
            item["reason"] = self._build_reason(recipe, per_meal_target, user)
            candidates.append(item)

        candidates.sort(key=lambda r: r["score"], reverse=True)
        return {
            "daily_targets": {k: round(v, 1) for k, v in daily_targets.items()},
            "per_meal_target": {k: round(v, 1) for k, v in per_meal_target.items()},
            "recommendations": candidates[:top_k],
        }


class MealRecommendationSystem:
    def __init__(self, model_path: str, recipes_path: str) -> None:
        self.model = UserNutritionModel()
        self.model.load(model_path)
        self.recommender = RecipeRecommender(recipes_path)

    def recommend(self, user: UserProfile, top_k: int = 5) -> Dict[str, Any]:
        daily_targets = self.model.predict_daily_targets(user)
        return self.recommender.recommend(user, daily_targets, top_k=top_k)


def ensure_model_exists(model_path: str, nutrition_csv_path: str, model_name: str = "random_forest") -> None:
    if os.path.exists(model_path):
        return
    model = UserNutritionModel(model_name=model_name)
    model.train(nutrition_csv_path)
    model.save(model_path)
