from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
# from pdfminer.high_level import extract_text
import importnb
sorted
import os
from chatbot import load_book,create_embedding,split_text,refining_result,chunks,pinecone_connection,upsert_data,index_,quering

app = Flask(__name__)
CORS(app)

splitted_text = []

@app.route('/')
def index():
    return render_template('index.html')

    
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    x = 3
    user_message = data.get('message')
    # Here, add your chatbot logic to respond to the user_message
    response = f"Echo: {user_message}"
    print('The user message is ',user_message)
    result = quering(user_message)
    #refining the obtained result
    refined_result = refining_result(result)
    return jsonify({'response': refined_result})
    


@app.route('/upload', methods=['POST'])
def upload_file():
    print('The route is hit ')
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        print('file is saved to the directory')
        #load the document
    document = load_book('C:/Users/Hp/OneDrive/Documents/Desktop/RAG/RAG/backend/uploads/')
        #split the document
    splitted_document = split_text(document) #make this splitted text global somehow 
    print('The document is splitted',len(splitted_document))
        #create the embeddings of the chunks 
    embedding_list = create_embedding(splitted_document)
    print('created the embedding')
        #make the pinecone connection 
    pinecone_connection()
        #formating the data to upsert 
    data_to_upsert = upsert_data(embedding_list,splitted_document)
    print('The data to upsert is ready')
        #store the chunks into pinecone index
    print(index)
    for ids_vectors_chunk in chunks(data_to_upsert, batch_size=200):
        index_.upsert(vectors=ids_vectors_chunk)
    print('The data is ready is usperted')
    print('The data is ready is usperted')
    print('The data is ready is usperted')
    return jsonify({'response': 'Data is ready to chat'})
        

    
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)

# the next step is to deploy this on aws 