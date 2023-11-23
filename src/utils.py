import openai
import numpy as np
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

meta_relations_dict = {
    "indication": ('drug', 'indication', 'disease'),
    "phenotype_protein": ('effect/phenotype', 'phenotype_protein', 'gene/protein'),
    "phenotype_phenotype": ('effect/phenotype', 'phenotype_phenotype', 'effect/phenotype'),
    "disease_phenotype_positive": ('disease', 'disease_phenotype_positive', 'effect/phenotype'),
    "disease_protein": ('disease', "disease_protein", 'gene/protein'),
    "disease_disease": ('disease', 'disease_disease', 'disease'),
    "drug_effect": ('drug', 'drug_effect', 'effect/phenotype'),
    "question_drug": ("question", "question_drug", "drug"),
    "question_disease": ("question", "question_disease", 'disease'),
    "question_effect/phenotype": ("question", "question_effect/phenotype", 'effect/phenotype'),
    "question_gene/protein": ("question", "question_gene/protein", 'gene/protein'),
    "question_correct_answer": ("question", "question_correct_answer", "answer"),
    "question_wrong_answer": ("question", "question_wrong_answer", "answer"),
    "answer_drug": ("answer", "answer_drug", "drug"),
    "answer_disease": ("answer", "answer_disease", 'disease'),
    "answer_effect/phenotype": ("answer", "answer_effect/phenotype", 'effect/phenotype'),
    "answer_gene/protein": ("answer", "answer_gene/protein", 'gene/protein')
}

relation_types = ["indication", "phenotype_protein", "phenotype_phenotype", "disease_phenotype_positive", "disease_protein", "disease_disease", "drug_effect",
                  "question_drug", "question_disease", "question_phenotype", "question_protein", "question_correct_answer",
                  "answer_drug", "answer_disease", "answer_phenotype", "answer_protein"]

node_types = ['question', 'drug', 'disease', 'effect/phenotype', 'gene/protein', 'answer']

metadata = (['question', 'drug', 'disease', 'effect/phenotype', 'gene/protein', 'answer'],
            [('drug', 'indication', 'disease'),
             ('disease', 'rev_indication', 'drug'),
             ('effect/phenotype', 'phenotype_protein', 'gene/protein'),
             ('gene/protein', 'rev_phenotype_protein', 'effect/phenotype'),
             ('effect/phenotype', 'phenotype_phenotype', 'effect/phenotype'),
             ('effect/phenotype', 'rev_phenotype_phenotype', 'effect/phenotype'),
             ('disease', 'disease_phenotype_positive', 'effect/phenotype'),
             ('effect/phenotype', 'rev_disease_phenotype_positive', 'disease'),
             ('disease', "disease_protein", 'gene/protein'),
             ('gene/protein', "rev_disease_protein", 'disease'),
             ('disease', 'disease_disease', 'disease'),
             ('disease', 'rev_disease_disease', 'disease'),
             ('drug', 'drug_effect', 'effect/phenotype'),
             ('effect/phenotype', 'rev_drug_effect', 'drug'),
             ("question", "question_drug", "drug"),
             ("drug", "rev_question_drug", "question"),
             ("question", "question_disease", 'disease'),
             ("disease", "rev_question_disease", 'question'),
             ("question", "question_effect/phenotype", 'effect/phenotype'),
             ("effect/phenotype", "rev_question_effect/phenotype", 'question'),
             ("question", "question_gene/protein", 'gene/protein'),
             ("gene/protein", "rev_question_gene/protein", 'question'),
             ("question", "question_correct_answer", "answer"),
             ("question", "question_wrong_answer", "answer"),
             ("answer", "rev_question_correct_answer", "question"),
             ("answer", "rev_question_wrong_answer", "question"),
             ("answer", "answer_drug", "drug"),
             ("drug", "rev_answer_drug", "answer"),
             ("answer", "answer_disease", 'disease'),
             ("disease", "rev_answer_disease", 'answer'),
             ("answer", "answer_effect/phenotype", 'effect/phenotype'),
             ("effect/phenotype", "rev_answer_effect/phenotype", 'answer'),
             ("answer", "answer_gene/protein", 'gene/protein'),
             ("gene/protein", "rev_answer_gene/protein", 'answer')])


###
# old version

meta_relations_dict_complete = {
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
    "anatomy_protein_absent": ('anatomy', 'anatomy_protein_absent', 'gene/protein'),
    "question_drug": ("question", "question_drug", "drug"),
    "question_disease": ("question", "question_disease", 'disease'),
    "question_phenotype": ("question", "question_phenotype", 'effect/phenotype'),
    "question_protein": ("question", "question_protein", 'gene/protein'),
    "question_answer": ("question", "question_answer", "answer"),
    "answer_question": ("answer", "answer_question", "question"),
    "answer_drug": ("answer", "answer_drug", "drug"),
    "answer_disease": ("answer", "answer_disease", 'disease'),
    "answer_phenotype": ("answer", "answer_phenotype", 'effect/phenotype'),
    "answer_protein": ("answer", "answer_protein", 'gene/protein')
}


def embed_text(text):
    """please implement this function according to your domain and use-case"""
    # self.context_node_embedding = self.model.encode([self.text])
    try:
        embeddings = openai.Embedding.create(input=[text], model="text-embedding-ada-002")['data'][0]['embedding']
        open_ai_embedding = np.reshape(np.asarray(embeddings), (-1, np.asarray(embeddings).shape[0]))
        return open_ai_embedding

    except Exception as e:
        print("Error: {}, String: {}".format(e, text))
