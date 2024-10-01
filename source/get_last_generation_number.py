import EspPipeML.esp_utilities as esp_utilities

h = esp_utilities.load_from_pickle('../results/nsga2/checkpoint/feature_selection/unsw-nb15_proposal_auc_history.pkl')
last_generation_number = len(h) - 1
print(f"The last generation was {last_generation_number}")
