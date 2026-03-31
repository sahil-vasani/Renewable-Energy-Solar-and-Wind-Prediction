# =========================================
# IMPORTS (CORRECT STRUCTURE)
# =========================================

# SOLAR
from src.data_preprocessing.solar_preprocessing import preprocess_solar
from src.feature_engineering.solar_features import feature_engineering_solar

# WIND
from src.data_preprocessing.wind_preprocessing import preprocess_wind
from src.feature_engineering.wind_features import feature_engineering_wind


# =========================================
# SOLAR PIPELINE
# =========================================
def run_solar_pipeline():
    print("\n===== SOLAR PIPELINE START =====")

    preprocess_solar()
    feature_engineering_solar()

    print("\nTraining Solar Model...")
    import src.models.solar_model   # ✅ runs automatically

    print("\n✅ SOLAR DONE")


# =========================================
# WIND PIPELINE
# =========================================
def run_wind_pipeline():
    print("\n===== WIND PIPELINE START =====")

    preprocess_wind()
    feature_engineering_wind()

    print("\nTraining Wind Model...")
    import src.models.wind_model   

    print("\n✅ WIND DONE")


# =========================================
# MAIN
# =========================================
def main():
    print("\n🚀 RUNNING FULL PIPELINE")

    run_solar_pipeline()
    run_wind_pipeline()

    print("\n🎉 ALL DONE")


if __name__ == "__main__":
    main()