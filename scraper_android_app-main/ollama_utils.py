import pandas as pd
from collections import Counter
from langchain.prompts import PromptTemplate
import ollama
from dotenv import load_dotenv
import os
import json
import nltk
from datetime import datetime

nltk.download('punkt')

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

def load_comments(json_file):
    """Load comments from a JSON file with multiple lines of JSON objects."""
    comments = []
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                comment = json.loads(line.strip())
                if 'content' in comment and 'date' in comment:
                    comments.append({
                        'content': comment['content'],
                        'date': comment['date']
                    })
    except FileNotFoundError:
        print(f"Erro: O arquivo {json_file} não foi encontrado.")
    except json.JSONDecodeError as e:
        print(f"Erro ao ler o arquivo {json_file}: {e}")
    except Exception as e:
        print(f"Erro ao ler o arquivo {json_file}: {e}")
    return comments

def create_prompt_template(case_of_use):
    templates = {
        "analise_sentimento": (
            "Analisar o sentimento dos seguintes comentários (positivo, negativo, neutro):\n{comments}. "
            "O resultado deve ser estritamente um JSON estruturado com os seguintes campos: "
            "'Tabela de Sentimentos' (um dicionário com a contagem de cada sentimento), "
            "'Comentários por Tópicos' (um dicionário com listas de comentários categorizados por sentimento e obrigatoriamente a data do comentário), "
            "e 'Análise para o Conselho Executivo' (um texto de no máximo 10 linhas). "
            "Não inclua nenhum texto adicional fora do JSON."
        )
    }
    return templates.get(case_of_use, "")

def save_results(results, prefix):
    """Save data to a JSON file with a unique name."""
    try:
        base_date = datetime.now().strftime("%d_%m_%Y")
        filename = f"{base_date}_{prefix}.json"
        counter = 1
        while os.path.exists(filename):
            filename = f"{base_date}_{counter}_{prefix}.json"
            counter += 1
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Resultados salvos em {filename}")
    except Exception as e:
        print(f"Erro ao salvar os resultados em {filename}: {e}")

def extract_json(content):
    """Extract JSON part from the content."""
    try:
        start_index = content.index('{')
        end_index = content.rindex('}') + 1
        json_content = content[start_index:end_index]
        return json_content
    except ValueError:
        return ""

def format_response(result):
    """Format the response to match the expected JSON structure."""
    sentiment_counts = Counter()
    comments_by_sentiment = {"Positivo": [], "Negativo": [], "Neutro": []}

    for sentiment, comments in result["Comentários por Tópicos"].items():
        if sentiment in comments_by_sentiment:
            sentiment_counts[sentiment] = len(comments)
            for comment in comments:
                comments_by_sentiment[sentiment].append({
                    "Data": comment.get("data", ""),
                    "Comentário": comment.get("comentario", "")
                })

    formatted_result = {
        "Tabela de Sentimentos": {
            "Positivo": sentiment_counts["Positivo"],
            "Negativo": sentiment_counts["Negativo"],
            "Neutro": sentiment_counts["Neutro"]
        },
        "Comentários por Tópicos": {
            "Positivo": comments_by_sentiment["Positivo"],
            "Negativo": comments_by_sentiment["Negativo"],
            "Neutro": comments_by_sentiment["Neutro"]
        },
        "Análise para o Conselho Executivo": result.get("Análise para o Conselho Executivo", "")
    }
    return formatted_result

def analyze_sentiment(comments):
    """Analyze sentiment of comments."""
    try:
        prompt_template = create_prompt_template('analise_sentimento')
        formatted_comments = "\n".join([f"{comment['date']} - {comment['content']}" for comment in comments])
        prompt = prompt_template.format(comments=formatted_comments)
        
        # Print the prompt for debugging
        print("Prompt being sent to the model:")
        print(prompt)
        
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        
        # Print the response for debugging
        print(response)
        
        # Extract and clean the content from the response
        content = response['message']['content']
        
        # Check if content is JSON or needs to be extracted
        json_content = extract_json(content)
        
        if not json_content:
            raise ValueError("Failed to extract JSON content from the response.")
        
        # Print the cleaned content for debugging
        print("Cleaned content:")
        print(json_content)
        
        # Parse the JSON response
        result = json.loads(json_content)
        
        # Format the response
        formatted_result = format_response(result)
        
        # Print the result for debugging
        print("Result to be saved:")
        print(json.dumps(formatted_result, ensure_ascii=False, indent=4))
        
        save_results(formatted_result, "android_analysis_sentiment")
    except json.JSONDecodeError as e:
        print(f"Erro ao analisar sentimentos: problema ao decodificar JSON - {e}")
    except ValueError as e:
        print(f"Erro ao analisar sentimentos: {e}")
    except Exception as e:
        print(f"Erro ao analisar sentimentos: {e}")

def print_header(header):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(header)
    print("=" * 80)

# Exemplo de uso
if __name__ == "__main__":
    print_header("Análise de Sentimento")
    json_file = "/mnt/data/14_06_2024_google_play_review.json"  # Substitua pelo nome do seu arquivo de comentários
    comments = load_comments(json_file)
    if comments:
        analyze_sentiment(comments)