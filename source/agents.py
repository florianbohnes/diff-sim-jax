class BaseControllerAgent:
    @staticmethod
    def get_action(x):
        """Passes the base action through without modification."""
        return x['action_planner']
