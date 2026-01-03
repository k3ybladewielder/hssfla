import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import SnowballStemmer
from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rouge_score import rouge_scorer

def preprocess_text(corpus, language="english"):
    # Instanciar o SnowballStemmer com base no idioma
    stemmer = SnowballStemmer(language)

    # Remove HTML tags
    corpus = re.sub(r'<.*?>', '', corpus)
    # Remove URLs
    corpus = re.sub(r'http\S+|www\S+|https\S+', '', corpus)

    # Remove quebras de linha
    corpus = corpus.replace('\n', '')
    corpus = re.sub(r'(?<=\S)\n(?=\S)', '', corpus)
    corpus = re.sub(r'(?<=\S)\n', '', corpus)

    # Remove espaços excessivos
    corpus = corpus.strip()

    # Converte o texto para minúsculas
    corpus = corpus.lower()

    # Tokeniza o texto em palavras
    words = word_tokenize(corpus)

    # Remove stopwords com base no idioma
    stop_words = set(stopwords.words(language))
    words = [word for word in words if word.lower() not in stop_words]

    # Aplica stemming com o SnowballStemmer no idioma especificado
    stemmed_words = [stemmer.stem(word) for word in words]

    # Junta as palavras novamente em uma string
    corpus = " ".join(stemmed_words)

    return corpus

def preprocess_text(corpus, language="english"):
    # Instanciar o SnowballStemmer com base no idioma
    stemmer = SnowballStemmer(language)

    # Remove HTML tags
    corpus = re.sub(r'<.*?>', '', corpus)

    # Remove URLs
    corpus = re.sub(r'http\S+|www\S+|https\S+', '', corpus)

    # Remove quebras de linha e espaços excessivos
    corpus = corpus.replace('\n', '').strip()

    # Converte o texto para minúsculas
    corpus = corpus.lower()

    # Tokeniza o texto em palavras
    words = word_tokenize(corpus)

    # Lista de caracteres indesejados
    #unwanted_chars = set([".", ",", "!", "?", "'", "`", "'", "'m", "'re", "'ve", "'ll", "'d", "``", ". "])
    unwanted_chars = set([".", ",", "!", "?", "'", "`", "'", "``", ". ", "''", "--", "'s", '(', ')', "»", ": "])
	
    # Remove stopwords e caracteres indesejados
    stop_words = set(stopwords.words(language))
    words = [word for word in words if word.lower() not in stop_words and word.lower() not in unwanted_chars]

    # Aplica stemming com o SnowballStemmer no idioma especificado
    stemmed_words = [stemmer.stem(word) for word in words]

    # Junta as palavras novamente em uma string
    corpus = " ".join(stemmed_words)

    return corpus
    
    
def preprocess_text_no_stem_stopwords(corpus, language="english"):
    """
    Função de pré-processamento de texto que remove HTML, URLs, quebras de linha, espaços excessivos
    e converte o texto para minúsculas, sem aplicar stemming ou remover stopwords.

    :param corpus: O texto a ser pré-processado.
    :param language: Idioma (padrão: inglês).
    :return: O texto pré-processado.
    """
    # Remove HTML tags
    corpus = re.sub(r'<.*?>', '', corpus)
    
    # Remove URLs
    corpus = re.sub(r'http\S+|www\S+|https\S+', '', corpus)

    # Remove quebras de linha
    corpus = corpus.replace('\n', '. ')

    # Remove espaços excessivos
    corpus = corpus.strip()

    # Converte o texto para minúsculas
    corpus = corpus.lower()

    # Tokeniza o texto em palavras (sem remover stopwords ou aplicar stemming)
    words = word_tokenize(corpus)

    # Junta as palavras novamente em uma string
    corpus = " ".join(words)

    return corpus
    
def get_tf_isf_matrix(sentences_list):
  vectorizer = CountVectorizer()
  count_vectorizer_matrix = vectorizer.fit_transform(sentences_list)

  # Obter a matriz TF (contagem de palavras)
  tf_matrix = count_vectorizer_matrix.toarray()

  # Número total de sentenças
  n_sentences = len(sentences_list)

  # Número de sentenças contendo cada termo (frequência de documentos)
  df = np.count_nonzero(tf_matrix, axis=0)

  # Calcular ISF: log(n_sentences / df)
  isf = np.log(n_sentences / df)

  # Calcular TF-ISF multiplicando TF por ISF
  tf_isf = tf_matrix * isf
  return tf_isf
  
  
def average_pooling(tf_isf, n_colunas_saida=1000):
    """
    Aplica o mean pooling em uma matriz TF-ISF, reduzindo o número de colunas 
    para um número desejado especificado.

    :param n_colunas_saida: Número de colunas desejadas após o pooling (padrão: 1000).
    :param tf_isf: Matriz TF-ISF que representa os dados de entrada, onde cada linha 
                   representa uma instância e cada coluna representa uma característica.
    
    :return: Uma nova matriz resultante após o pooling, com o número de colunas 
             reduzido conforme especificado.
    
    A função calcula o tamanho do grupo necessário para atingir o número de colunas 
    de saída e aplica padding se necessário. O pooling é realizado pela média dos 
    elementos em cada grupo. 
    """

    n_cols_matrix = tf_isf.shape[1]  # Número de colunas na matriz original

    # Calcular o tamanho do grupo necessário para atingir n_colunas_saida
    n_grupos = (n_cols_matrix + n_colunas_saida - 1) // n_colunas_saida  # Tamanho do grupo para pooling

    # Padding para lidar com colunas restantes, se houver
    padding = n_grupos * n_colunas_saida - n_cols_matrix  # Calcular o padding necessário
    if padding > 0:
        tf_isf = np.pad(tf_isf, ((0, 0), (0, padding)), mode='constant', constant_values=0)  # Padding com zeros

    # Reshape e pooling (média)
    tf_isf_pooled = tf_isf.reshape(tf_isf.shape[0], -1, n_grupos).mean(axis=2)
    return tf_isf_pooled
    
def max_pooling(tf_isf, n_colunas_saida=1000):
    """
    Aplica o max pooling em uma matriz TF-ISF, reduzindo o número de colunas 
    para um número desejado especificado.

    :param n_colunas_saida: Número de colunas desejadas após o pooling (padrão: 1000).
    :param tf_isf: Matriz TF-ISF que representa os dados de entrada, onde cada linha 
                   representa uma instância e cada coluna representa uma característica.
    
    :return: Uma nova matriz resultante após o pooling, com o número de colunas 
             reduzido conforme especificado.
    
    A função calcula o tamanho do grupo necessário para atingir o número de colunas 
    de saída e aplica padding se necessário. O pooling é realizado pegando o valor 
    máximo dos elementos em cada grupo.
    """

    n_cols_matrix = tf_isf.shape[1]  # Número de colunas na matriz original

    # Calcular o tamanho do grupo necessário para atingir n_colunas_saida
    n_grupos = (n_cols_matrix + n_colunas_saida - 1) // n_colunas_saida  # Tamanho do grupo para pooling

    # Padding para lidar com colunas restantes, se houver
    padding = n_grupos * n_colunas_saida - n_cols_matrix  # Calcular o padding necessário
    if padding > 0:
        tf_isf = np.pad(tf_isf, ((0, 0), (0, padding)), mode='constant', constant_values=0)  # Padding com zeros

    # Reshape e pooling (máximo)
    tf_isf_pooled = tf_isf.reshape(tf_isf.shape[0], -1, n_grupos).max(axis=2)
    return tf_isf_pooled
    
# Função para gerar bigramas a partir de uma palavra
def gerar_bigramas(palavra):
    return list(ngrams(palavra, 2))

# Função para criar o mapeamento entre bigramas e palavras originais
def criar_mapeamento_palavras(texto_original, texto_stemmed):
    mapeamento_bigramas = {}
    
    for palavra_original, palavra_stemmed in zip(texto_original, texto_stemmed):
        bigramas = gerar_bigramas(palavra_stemmed)
        for bigrama in bigramas:
            if bigrama not in mapeamento_bigramas:
                mapeamento_bigramas[bigrama] = palavra_original
                
    return mapeamento_bigramas

# Função para reverter o stemming usando o mapeamento de bigramas
def reverter_stemming_frase(frase_stemmatizada, mapeamento_bigramas):
    palavras_revertidas = []
    
    for palavra_stemmatizada in frase_stemmatizada:
        bigramas = gerar_bigramas(palavra_stemmatizada)
        palavras_reconstruidas = []
        
        for bigrama in bigramas:
            if bigrama in mapeamento_bigramas:
                palavras_reconstruidas.append(mapeamento_bigramas[bigrama])
        
        # Se encontramos uma palavra associada, usamos ela, caso contrário mantemos a stemmatizada
        if palavras_reconstruidas:
            palavras_revertidas.append(palavras_reconstruidas[0])
        else:
            palavras_revertidas.append(palavra_stemmatizada)
    
    return palavras_revertidas
    
def process_corpus(corpus, use_max_pooling=False): #, n_cols=250):
    # Primeira etapa: dividir o texto em frases
    corpus_list = sent_tokenize(corpus)

    # Segunda etapa: processar cada frase individualmente
    original_sentences_list = [preprocess_text_no_stem_stopwords(sentence) for sentence in corpus_list]
    preprocessed_sentences_list = [preprocess_text(sentence) for sentence in corpus_list]

    tf_isf = get_tf_isf_matrix(preprocessed_sentences_list)
    embeddings = tf_isf # utilizando a matriz tf-isf max pooled criada acima

    if use_max_pooling:
        n_cols = tf_isf.shape[1] // 3 # diminuindo a matriz original por 1/3 dela ou qualquer inteiro
        #n_cols = n_cols
        tf_isf_pooled = max_pooling(tf_isf, n_colunas_saida=n_cols)
        embeddings = tf_isf_pooled # utilizando a matriz tf-isf max pooled criada acima

    return embeddings, preprocessed_sentences_list
    
def process_corpus_with_stemming(corpus, use_max_pooling=False): #, n_cols=250):
    # Primeira etapa: dividir o texto em frases
    corpus_list = sent_tokenize(corpus)

    # Segunda etapa: processar cada frase individualmente
    original_sentences_list = [preprocess_text(sentence) for sentence in corpus_list]
    preprocessed_sentences_list = [preprocess_text(sentence) for sentence in corpus_list]

    tf_isf = get_tf_isf_matrix(preprocessed_sentences_list)
    embeddings = tf_isf # utilizando a matriz tf-isf max pooled criada acima

    if use_max_pooling:
        n_cols = tf_isf.shape[1] // 3 # diminuindo a matriz original por 1/3 dela ou qualquer inteiro
        #n_cols = n_cols
        tf_isf_pooled = max_pooling(tf_isf, n_colunas_saida=n_cols)
        embeddings = tf_isf_pooled # utilizando a matriz tf-isf max pooled criada acima

    return embeddings, preprocessed_sentences_list
    
    
# Inicializando o calculador de ROUGE fora da função
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

# Função para calcular as métricas de ROUGE
def calcular_metricas_rouge(row):
    # Obtendo o summary de referência (ground truth)
    reference_summary = row["reference"]
    #reference_summary = " ".join(reference_summary)
    reference_summary = reference_summary.replace("[\n", " ").replace("\n", " ").replace("[", "").replace("`", "").replace(" . ", ". ").replace("''", "").replace(" ,", ",").replace(" .", ".").replace("..", ".").replace("  ", " ").strip().lower()
    reference_summary = preprocess_text(reference_summary)
    
    # Obtendo o resumo extraído (candidato)
    candidate_summary = row["summary"]
    candidate_summary = candidate_summary.replace("\n", " ").replace("[", "").replace("`", "").replace(" . ", ". ").replace("''", "").replace(" ,", ",").replace(" .", ".").replace("..", ".").replace("  ", " ").strip().lower()
    candidate_summary = preprocess_text(candidate_summary)  # Adicionando pré-processamento para o resumo candidato
    
    # Calculando ROUGE entre o summary (referência) e o summary extraído (candidato)
    scores_ = scorer.score(reference_summary, candidate_summary)
    
    # Armazenando os valores de ROUGE-1 e ROUGE-2
    rouge1_precision = scores_['rouge1'].precision
    rouge2_precision = scores_['rouge2'].precision
    
    rouge1_recall = scores_['rouge1'].recall
    rouge2_recall = scores_['rouge2'].recall
    
    
    rouge1_fmeasure = scores_['rouge1'].fmeasure
    rouge2_fmeasure = scores_['rouge2'].fmeasure
    
    # Retornando as métricas calculadas
    return pd.Series({
        "rouge1_precision": rouge1_precision,
        "rouge2_precision": rouge2_precision,
        
        "rouge1_recall": rouge1_recall,
        "rouge2_recall": rouge2_recall,
        
        "rouge1_fmeasure": rouge1_fmeasure,
        "rouge2_fmeasure": rouge2_fmeasure
    })

