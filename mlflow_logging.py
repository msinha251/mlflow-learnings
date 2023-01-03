import mlflow
import mlflow.sklearn

def log_model(model, X_train, X_test, y_train, y_test, model_name, model_type):
    '''
    This function takes a trained model, training and test data, a model name, and a model type (either "regression" or "classification") as input. 
    It logs the model and training parameters, trains the model, makes predictions on the test set, and logs various evaluation metrics depending on the model type. 
    It also logs any artifacts (such as plots) that you want to include, and ends the run.
    '''
    with mlflow.start_run() as run:
        # log model and training parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", model_type)
        mlflow.sklearn.log_model(model, "model")

        # train model
        model.fit(X_train, y_train)

        # make predictions on test set
        y_pred = model.predict(X_test)

    # log evaluation metrics
    if model_type == "regression":
        mlflow.log_metric("mean_absolute_error", metrics.mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("mean_squared_error", metrics.mean_squared_error(y_test, y_pred))
        mlflow.log_metric("root_mean_squared_error", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        mlflow.log_metric("r2_score", metrics.r2_score(y_test, y_pred))
    elif model_type == "classification":
        mlflow.log_metric("accuracy", metrics.accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", metrics.precision_score(y_test, y_pred))
        mlflow.log_metric("recall", metrics.recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", metrics.f1_score(y_test, y_pred))

    # log artifacts (e.g. plots)
    if model_type == "regression":
        mlflow.log_artifact(plot_prediction_results(y_test, y_pred), "plot")
    elif model_type == "classification":
        mlflow.log_artifact(plot_confusion_matrix(y_test, y_pred), "plot")

    # end run
    mlflow.end_run()


if __name__ == "__main__":
    # # load data
    # X_train, X_test, y_train, y_test = load_data()

    # # train model
    # model = train_model(X_train, y_train)

    # log model
    log_model(model, X_train, X_test, y_train, y_test, "Random Forest Regressor", "regression")
