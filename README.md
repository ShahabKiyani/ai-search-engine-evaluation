# 1. Clone repo
git clone https://github.com/ShahabKiyani/ai-search-engine-evaluation.git
cd ai-search-engine-evaluation

# 2. Create virtual environment in terminal
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies (run in terminal after you see (venv) in terminal bar)
pip install -r requirements.txt


run this in terminal:
1.) python download_data.py

2.)python build_nq_index.py


finally, run the app using this command:
streamlit run main2_streamlit.py
