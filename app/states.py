from FeatureCloud.app.engine.app import AppState, app_state, Role
import random_forest
from FeatureCloud.app.engine.app import AppState, app_state


@app_state("initial")
class ExecuteState(AppState):

    def register(self):
        self.register_transition("terminal", Role.BOTH)

    def run(self):
        random_forest.main()
        return "terminal"
