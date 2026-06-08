class Evaluator:

    def evaluate(self, recommendations):

        print(f"Generated {len(recommendations)} recommendations")

        return {
            "recommendation_count": len(recommendations)
        }