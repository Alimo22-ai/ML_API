from meal_model import UserNutritionModel


if __name__ == "__main__":
    model = UserNutritionModel()
    model.train("nutrition_dataset.csv")
    model.save("nutrition_model.joblib")
    print("Saved model to nutrition_model.joblib")
