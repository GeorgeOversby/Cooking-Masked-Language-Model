from src.data_processing import remove_split_ingredients


def test_remove_split_ingredients():
    ingredients = {"sherry_wine", "sherry_wine_vinegar", "cheese", "cheddar_cheese"}

    assert remove_split_ingredients(ingredients) == {
        "sherry_wine_vinegar",
        "cheddar_cheese",
    }
