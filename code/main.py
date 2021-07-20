import warnings
import pandas as pd
import createLDA_featurevector, doc_clustering, fineGrainLocation as fgl

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    # This can be changed when we want to check other users
    location_data = pd.read_csv("../Resources/Parsed_Data/user1.csv", parse_dates=['Timestamp'])
    multiple_Loc = fgl.parse_data(location_data)
    fgl.set_labels(multiple_Loc, location_data)
    fgl.convert_to_date(location_data)
    df = fgl.fine_grain_location(location_data)
    df.to_csv('fineGrainLocation.csv', index=False)
    list = createLDA_featurevector.makeDayDescriptorList(df)
    pd.DataFrame(list).to_csv('makeDayDescriptor.csv', index=False)
    list_of_docs = createLDA_featurevector.makeDocuments(list)
    featureVec, featurevec_df = createLDA_featurevector.modeling(list_of_docs)
    doc_clustering.clustering(featureVec, featurevec_df)


if __name__ == "__main__":
    main()
