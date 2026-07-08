from sklearn.base import BaseEstimator, TransformerMixin


class SalaryFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.education_map = {
            "High School": 0,
            "Bachelor": 1,
            "Master": 2,
            "PhD": 3
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # Education ordinal encoding
        X["EducationLevel"] = (
            X["EducationLevel"]
            .map(self.education_map)
        )

        # Department encoding
        departments = [
            "Engineering",
            "Finance",
            "HR",
            "Marketing",
            "Operations",
            "Sales"
        ]

        for dept in departments:
            X[f"{dept}Department"] = (
                X["Department"] == dept
            ).astype(int)

        # Remove original department
        X.drop(
            "Department",
            axis=1,
            inplace=True
        )

        # Feature engineering
        X["EducationExperienceImpact"] = (
            X["EducationLevel"] *
            X["YearsExperience"]
        )

        X["EducationCompanyLoyalty"] = (
            X["EducationLevel"] *
            X["YearsAtCompany"]
        )

        X["EducationPerformanceBoost"] = (
            X["EducationLevel"] *
            X["PerformanceRating"]
        )

        return X