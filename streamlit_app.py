import streamlit as st
import pandas as pd
import numpy as np
import os
import csv
import requests
import scipy.sparse as sparse
from lightfm.data import Dataset
from lightfm import LightFM

data_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"


def make_clickable(val):
    return '<a target="_blank" href="{}">Goodreads</a>'.format(val)

def path_to_image_html(path):
    return '<img src="'+ path + '" width="50" >'

def _download(url: str, dest_path: str):

    req = requests.get(url, stream=True)
    req.raise_for_status()

    with open(dest_path, "wb") as fd:
        for chunk in req.iter_content(chunk_size=2 ** 20):
            fd.write(chunk)

def get_data(string):

    dat_string =  string + ".csv"
    ratings_url = data_url + dat_string
    dest_path = "data/" + string + ".csv"

    if not os.path.exists("data"):
        os.makedirs("data")
        
        _download(ratings_url, dest_path)
    else:
        if not os.path.exists(dest_path):
            _download(ratings_url, dest_path)
    
    with open(dest_path, "r") as dat:
        dat = [x for x in csv.DictReader(dat)] 
        return dat

def get_ratings():

    return get_data("ratings"), get_data("books")

def give_titles(book_features):
    title_list = {"book_id" : [x['book_id'] for x in book_features], 
                  "title" : [x['title'] for x in book_features], 
                  "img_url": [x['image_url'] for x in book_features]}
    title_frame = pd.DataFrame(data = title_list)
    return title_frame


st.title("Book Recommender [Only till 2017, Sorry..]")

st.write("Hit enter without text in the book-input to exit rating process!\n")


if 'rate_data' not in st.session_state:
    st.session_state.rate_data = get_ratings()[0]

if 'book_data' not in st.session_state:
    st.session_state.book_data = get_ratings()[1]

ratings, book_features = st.session_state.rate_data, st.session_state.book_data

if 'title_frame' not in st.session_state:
    st.session_state['title_frame'] = give_titles(book_features) 

title_frame = st.session_state.title_frame


def string_list(substring):
    if substring == "":
        return pd.DataFrame()
    substring = substring.lower() 
    seen = []
    for item in title_frame.itertuples(index = False):
        if (item.title.lower()).find(substring) == -1:
            continue
        else: 
            seen.append(item)  

    if len(seen) == 0:
        return pd.DataFrame() 
    return pd.DataFrame(data = seen)

def inv_dict(dicti):
    return dict(zip(dicti.values(), dicti.keys()))

def scale_weights(weights, interactions):
    count_treat = interactions.sum(axis = 0)[0,]
    C = np.squeeze(np.asarray(count_treat))

    inv_pscore = np.power(C, -(0.45)*np.ones(len(C)))
    adj_score_mat = weights.multiply(sparse.csr_array(inv_pscore))
    mx_wgt = adj_score_mat.max()
    max_norm_inv = np.power(mx_wgt, -1)
    
    scaled_weights = adj_score_mat.tocoo()*(max_norm_inv*5)
    return scaled_weights

def dataset_build(ratings, curr_ratings, book_features):

    dataset = Dataset()
    dataset.fit((x['user_id'] for x in ratings+curr_ratings),
                (x['book_id'] for x in ratings+curr_ratings),
                (x['rating'] for x in ratings+curr_ratings))  

    dataset.fit_partial(items=(x['book_id'] for x in book_features),
                        item_features=(x['authors'] for x in book_features))

    item_features = dataset.build_item_features(((x['book_id']), [x['authors']])
                                                for x in book_features)

    return dataset, item_features

def return_interactions(dataset, curr_ratings):
    (interactions, weights) = dataset.build_interactions((x['user_id'], x['book_id'], int(x['rating'])) 
                                                      for x in ratings+curr_ratings)
    return (interactions, weights)

def get_predictions(dataset, user_ids, model, weights, book_features):

    cont = st.container(border=True)
    num_users, num_items = weights.shape

    weight = weights.tocsr()
    mapping = dataset.mapping()
    inv_mapping = [inv_dict(x) for x in mapping]
        
    for user_id in user_ids:
        user_id = str(user_id)
        in_map = mapping[0][user_id]
    
        scores = model.predict(mapping[0][user_id], np.arange(num_items))

        known_read = [{"title": book_features[int(inv_mapping[2][x])-1]['title'], 
                       "rating" : weight[in_map ,x],
                       "img" : book_features[int(inv_mapping[2][x])-1]['image_url']} 
                       for x in range(num_items) if weight[in_map ,x] > 0]
        
        top_items = [{"title": book_features[int(inv_mapping[2][x])-1]['title'],
                       "img": book_features[int(inv_mapping[2][x])-1]['image_url']}
                       for x in np.argsort(-scores)] 

        known_read.sort(key=lambda dict: dict["rating"], reverse = True)
        known_reads = [x['title'] for x in known_read]
        known_read_set = set(known_reads)
        rec_list = []
        count = 0
        for x in top_items:
                if x['title'] not in known_read_set:
                    rec_list.append(x)
                    count+=1
                if count == 5:
                    break
        


        st.session_state.known_df = pd.DataFrame(data = known_read).head(5)
        st.session_state.rec_df = pd.DataFrame(data= rec_list).head(5)



def run_model(curr_ratings):

    if len(curr_ratings) == 0:
        cut_frame = title_frame.head(5)
        display_data = cut_frame.loc[:, search_list.columns != 'book_id']
        display_data['img'] = display_data.apply( lambda x: path_to_image_html(x['img_url']), axis = 1 )
        final = display_data.loc[:, display_data.columns != 'img_url']
        error.write(final.to_html(escape=False), unsafe_allow_html=True)

    else:
        dataset, item_features = dataset_build(ratings, curr_ratings, book_features)
        interactions, weights = return_interactions(dataset, curr_ratings)

        weights = weights.tocsr().tocoo()
        interactions = interactions.tocsr().tocoo()

        scaled_weights = scale_weights(weights, interactions)

        model = LightFM(no_components = 20, loss='warp')
        model.fit(interactions, item_features = item_features, sample_weight = scaled_weights, epochs = 25, num_threads = 7)

        get_predictions(dataset, [-1], model, weights, book_features)


def error_check():
    book_name = st.session_state.bn
    if len(book_name) == 0 or len(book_name) < 4:
        st.session_state.bn = ""
        st.session_state.bcb = False

    else:
        st.session_state.bcb = True

def submit_entry():
    search_list = string_list(st.session_state.bn)
    index_choice = int(st.session_state.index)
    rating = st.session_state.rat

    if index_choice > len(search_list.index) - 1 or index_choice < 0:
        st.session_state.bcr = False
        del st.session_state['index']
        del st.session_state['rat']

    elif rating.isdigit() == False or int(rating) > 5 or int(rating) < 1:
        st.session_state.bcr = False
        del st.session_state['index']
        del st.session_state['rat']

    else:
        st.session_state.bcr = True
        row = search_list.iloc[int(index_choice)]
        book_id, title = row.loc['book_id'], row['title']
        
        st.session_state.list.append({'user_id': '-1', 'book_id': book_id, 'rating': int(rating)}) 
        error.text(title)
        st.session_state['bn'] = ""

def give_recs():
        curr_ratings = st.session_state.list
        if len(curr_ratings) == 0:
            st.write()
        run_model(curr_ratings)




if 'list' not in st.session_state:    
    st.session_state['list'] = []

if 'bcb' not in st.session_state:
    st.session_state['bcb'] = False

if 'bcr' not in st.session_state:
    st.session_state['bcr'] = False


error = st.empty()

book_name = st.text_input("Enter book to rate (You could enter a substring at least 4 characters):", key = 'bn', value = "", on_change = error_check)

if st.session_state.bcb and st.session_state.bn != "":


    search_list = string_list(st.session_state.bn)

    if search_list.empty:
        error.write("Can't seem to find it, matey!")

    else:
        display_data = search_list.loc[:, search_list.columns != 'book_id']
        display_data['img'] = display_data.apply( lambda x: path_to_image_html(x['img_url']), axis = 1 )
        final = display_data.loc[:, display_data.columns != 'img_url']
        
        with error.container():

            if st.session_state.bcr:
                st.write("Please fill the form using the index of the book you want to rate")
            else:
                st.write("Messed the ratings, matey!")

            st.write(final.to_html(escape=False), unsafe_allow_html=True)

elif st.session_state.bn == "":
    error.text("Enter a book name matey!")

else:
    with error.container():
        st.write("You definitely messed up, I think. Or you haven't tried the site yet")


with st.form(key = "Book_Entry", clear_on_submit=True, border=True):

    index = st.text_input("Enter index of book to rate from above", key = 'index')
    
    rating = st.text_input("Enter rating for the book", key = 'rat')

    submitted = st.form_submit_button("Submit Entry", on_click = submit_entry)

        
model_runcheck = st.checkbox("Are you done rating?", key = 'mc', on_change=give_recs)

cont = st.empty()

if st.session_state.mc == True:
   
    rec_df = st.session_state.rec_df
    known_df = st.session_state.known_df
    
    
    display_data_rec = rec_df.loc[:, rec_df.columns != 'book_id']
    display_data_rec['image'] = display_data_rec.apply( lambda x: path_to_image_html(x['img']), axis = 1 )

    display_data_known = known_df.loc[:, known_df.columns != 'book_id']
    display_data_known['image'] = display_data_known.apply( lambda x: path_to_image_html(x['img']), axis = 1 )

        

    with cont.container():
        st.write("Results for the user")
        st.write("     Known positives:")
        st.write(display_data_known.loc[:, display_data_known.columns != 'img'].to_html(escape=False), unsafe_allow_html=True)

        st.write("     Recommended:")
        st.write(display_data_rec.loc[:, display_data_rec.columns != 'img'].to_html(escape=False), unsafe_allow_html=True)
