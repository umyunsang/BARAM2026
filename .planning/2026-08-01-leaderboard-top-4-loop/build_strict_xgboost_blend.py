"""Build the Q2-selected chronology-safe XGBoost blend."""

import build_strict_dart_blend as builder


def main() -> None:
    builder.BOOSTER_ID = "M115_XGBOOST"
    builder.CANDIDATE_ID = "M116_STRICT_XGBOOST_BLEND"
    builder.SELECTED_ITERATION = 100
    builder.main()


if __name__ == "__main__":
    main()
