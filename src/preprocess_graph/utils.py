
meta_relation_dict = {
    "protein_protein": ('gene/protein', 'protein_protein', 'gene/protein'),
    "drug_protein": ('drug', "drug_protein", 'gene/protein'),
    "contraindication": ('drug', 'contraindication', 'disease'),
    "indication": ('drug', 'indication', 'disease'),
    "off_label_use": ('drug', 'off_label_use', 'disease'),
    "drug_drug": ('drug', 'drug_drug', 'drug'),
    "phenotype_protein": ('effect/phenotype', 'phenotype_protein', 'gene/protein'),
    "phenotype_phenotype": ('effect/phenotype', 'phenotype_phenotype', 'effect/phenotype'),
    "disease_phenotype_negative": ('disease', 'disease_phenotype_negative', 'effect/phenotype'),
    "disease_phenotype_positive": ('disease', 'disease_phenotype_positive', 'effect/phenotype'),
    "disease_protein": ('disease', "disease_protein", 'gene/protein'),
    "disease_disease": ('disease', 'disease_disease', 'disease'),
    "drug_effect": ('drug', 'drug_effect', 'effect/phenotype'),
    "bioprocess_bioprocess": ('biological_process', 'bioprocess_bioprocess', 'biological_process'),
    "molfunc_molfunc": ('molecular_function', 'molfunc_molfunc', 'molecular_function'),
    "cellcomp_cellcomp": ('cellular_component', 'cellcomp_cellcomp', 'cellular_component'),
    "molfunc_protein": ('molecular_function', 'molfunc_protein', 'gene/protein'),
    "cellcomp_protein": ('cellular_component, cellcomp_protein', 'gene/protein'),
    "bioprocess_protein": ('biological_process', "bioprocess_protein", 'gene/protein'),
    "exposure_protein": ('exposure', 'exposure_protein', 'gene/protein'),
    "exposure_disease": ('exposure', 'exposure_disease', 'disease'),
    "exposure_exposure": ('exposure', 'exposure_exposure', 'exposure'),
    "exposure_bioprocess": ('exposure', 'exposure_bioprocess', 'biological_process'),
    "exposure_molfunc": ('exposure', 'exposure_molfunc', 'molecular_function'),
    "exposure_cellcomp": ('exposure', 'exposure_cellcomp', 'cellular_component'),
    "pathway_pathway": ('pathway', 'pathway_pathway', 'pathway'),
    "pathway_protein": ('pathway', 'pathway_protein', 'gene/protein'),
    "anatomy_anatomy": ('anatomy', 'anatomy_anatomy', 'anatomy'),
    "anatomy_protein_present": ('anatomy', 'anatomy_protein_present', 'gene/protein'),
    "anatomy_protein_absent": ('anatomy', 'anatomy_protein_absent', 'gene/protein')}

relation_types = ["indication", "phenotype_protein", "phenotype_phenotype", "disease_phenotype_positive",
                  "disease_protein", "disease_disease", "drug_effect", "question_answer", "question_drug", "question_disease", "question_phenotype"]

node_types = ['question', 'drug', 'disease', 'effect/phenotype', 'gene/protein', 'answer']
