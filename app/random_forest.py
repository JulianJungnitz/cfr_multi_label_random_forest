# %%
import yaml
import logging
import utils as utils
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split


def execute_query(query):
    with driver.session(database=utils.NEO4J_DB) as session:
        try:
            logger.info(query)
            res = session.run(query).data()
            logger.info(f"Query returned {len(res)} records")
            return res
        except Exception as e:
            logger.error(f"Error: {e}")
            return None
        finally:
            session.close()


def get_binarizer(query, key):
    data = execute_query(query)
    data = [[record[key]] for record in data]
    binarizer = MultiLabelBinarizer()
    binarizer.fit_transform(data)
    return binarizer


def get_all_binaries():
    pheno_query = """
    MATCH (:Biological_sample)-[:HAS_PHENOTYPE]->(ph:Phenotype)
    RETURN DISTINCT ph.id AS phenotype
    """
    gene_query = """
    MATCH (:Biological_sample)-[:HAS_DAMAGE]->(g:Gene)
    RETURN DISTINCT g.id AS genes
    """

    pheno_binarizer = get_binarizer(pheno_query, "phenotype")
    gene_binarizer = get_binarizer(gene_query, "genes")
    return pheno_binarizer, gene_binarizer


def get_full_data_set(limit=None):
    query = """
    MATCH (bs:Biological_sample)-[:HAS_DISEASE]->(d:Disease)
    Optional MATCH (bs)-[:HAS_PHENOTYPE]->(ph:Phenotype)
    Optional MATCH (bs)-[:HAS_DAMAGE]->(g:Gene)
    RETURN ID(bs) as subject_id, collect(distinct ph.id) AS phenotypes, collect(distinct g.id) AS genes, collect(distinct d.id) as diseases, collect(distinct d.name) AS disease_names
    """
    if limit:
        query += f" LIMIT {limit}"
    data = execute_query(query)
    return data


def filter_with_binarizer(df, pheno_binarizer, gene_binarizer):

    pheno_features = pheno_binarizer.transform(df["phenotypes"])
    pheno_df = pd.DataFrame(pheno_features, columns=pheno_binarizer.classes_)

    gene_features = gene_binarizer.transform(df["genes"])
    gene_df = pd.DataFrame(gene_features, columns=gene_binarizer.classes_)

    df_final = pd.concat([df.reset_index(drop=True), pheno_df, gene_df], axis=1)
    df_final = df_final.drop(
        columns=["phenotypes", "genes", "proteins"], errors="ignore"
    )
    return df_final


def duplicate_columns_with_multiple_diseases(df):
    df_exploded = df.explode(["diseases", "disease_names"]).reset_index(drop=True)
    logger.info(
        f"DataFrame before exploding: {len(df)} rows. Dataframe after exploded: {len(df_exploded)} rows after exploding."
    )
    return df_exploded


def get_features_and_labels(df):
    features = df.drop(columns=["subject_id", "diseases", "disease_names"])
    labels = df["disease_names"].apply(
        lambda x: (
            0
            if (isinstance(x, list) and len(x) == 1 and x[0] == "control")
            else 1
        )
    )
    return features, labels


def train_classifier(features, labels):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    logger.info("Training classifier")
    clf.fit(features, labels)
    logger.info("Classifier trained")
    return clf


def split_train_test(
    df,
):
    df, control = train_test_split(df, test_size=0.2, random_state=1)
    return df, control


def predict_on_control(features, clf):
    y_pred = clf.predict(features)
    return y_pred


def save_to_csv(subject_ids, disease, filename):

    df = pd.DataFrame({"subject_id": subject_ids, "disease": disease})
    df.to_csv(filename, index=False)


def transform_data_into_features_and_labels(
    data, pheno_binarizer, gene_binarizer,
):
    df_final = filter_with_binarizer(data, pheno_binarizer, gene_binarizer)
    # df_final = duplicate_columns_with_multiple_diseases(df_final)
    features, labels = get_features_and_labels(df_final)
    return features, labels


def main():
    global logger
    global driver
    logger = utils.create_logger()

    logger.info("Starting the program")
    driver = utils.connect_to_neo4j()

    config = utils.read_config()

    pheno_binarizer, gene_binarizer = get_all_binaries()

    data = get_full_data_set()
    df = pd.DataFrame(data)

    # logger.info(f"Dataframe values of diseases: {df['disease_names']}")

    df, control = split_train_test(df)

    features, labels = transform_data_into_features_and_labels(
        df, pheno_binarizer, gene_binarizer, 
    )

    clf = train_classifier(features, labels)

    control_features, control_labels = transform_data_into_features_and_labels(
        control, pheno_binarizer, gene_binarizer
    )

    y_pred = predict_on_control(control_features, clf)

    save_to_csv(control["subject_id"], y_pred, "./predictions.csv")

    logger.info("----------------------------------------")
    logger.info(f"Training Shapes: {features.shape}, {labels.shape} ")
    logger.info(f"Shapes Control: {control_features.shape}, {control_labels.shape} ")
    logger.info(f"{metrics.classification_report(control_labels, y_pred)}")

    


    # error werfen wenn control mit anderer disease in der db ist

    # check clasifier specifications


if __name__ == "__main__":
    main()

# %%
