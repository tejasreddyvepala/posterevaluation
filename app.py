import os
import configparser
import cv2
import re
import requests
import numpy as np
import pytesseract
import language_tool_python
from PIL import ImageFont, ImageDraw, Image
import werkzeug.utils
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_file
import pdfkit
from flask import Response
import base64
from transformers import pipeline
import spacy
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
import openai


# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

openai.api_key = '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'

app = Flask(__name__)

# Initialize the ConfigParser and read the config file
config = configparser.ConfigParser()
config.read('config.ini')

# Flask settings
app.secret_key = config.get('flask', 'secret_key')
app.debug = config.getboolean('flask', 'debug')

# OpenAI settings
openai.api_key = config.get('openai', 'api_key')

# Upload settings
UPLOAD_FOLDER = config.get('upload', 'folder')
ALLOWED_EXTENSIONS = set(config.get('upload', 'allowed_extensions').split(', '))

# PDFKit settings
pdfkit_config = pdfkit.configuration(wkhtmltopdf=config.get('pdfkit', 'wkhtmltopdf_path'))

# Known fonts
KNOWN_FONTS = config.get('fonts', 'known_fonts').split(', ')

# Language tool
language = config.get('language_tool', 'language')

# Pytesseract custom config
custom_config = config.get('pytesseract', 'custom_config')

# spaCy model
nlp = spacy.load(config.get('spacy', 'model'))

# Educational keywords
educational_keywords = config.get('educational_keywords', 'keywords').split(', ')


pdfkit_config = pdfkit.configuration(wkhtmltopdf='/usr/local/bin/wkhtmltopdf')
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
KNOWN_FONTS = ["Arial", "Times New Roman", "Verdana", "Tahoma", "Calibri"]

def extract_references(text):
    url_pattern = r'\b(?:https?://)?(?:(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.[a-zA-Z]{2,6}\b/?)'
    return re.findall(url_pattern, text)

def get_wordnet_meaning(word):
    # Use WordNet to get meanings for a word
    synsets = wordnet.synsets(word)
    if synsets:
        return synsets[0].definition()
    return None

def summarize_text(text_to_summarize, max_tokens=100):
    
    prompt_text = """
    Please provide a detailed summary and analysis of the following content:
    {}

    Include information about the study's aims, methods, results, and conclusions.
    """.format(text_to_summarize)

    response = openai.Completion.create(
    engine="text-davinci-003", # or the latest available engine
    prompt=prompt_text,
    max_tokens=600 # Increased token limit for a more detailed response
    )

    return response.choices[0].text.strip()

def find_charts(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adjust the Gaussian blur parameters as needed
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduced kernel size for less blurring

    # Adjust the Canny edge detection parameters as needed
    edges = cv2.Canny(blurred, 50, 150)  # Adjusted thresholds

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Adjust the minimum contour area as needed
    min_contour_area = 500  # Reduced area for potentially smaller charts

    chart_images = []

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # Create a mask for the current chart
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

            # Extract the current chart using the mask
            chart = cv2.bitwise_and(img, img, mask=mask)

            # Find the bounding box around the chart and crop it
            x, y, w, h = cv2.boundingRect(contour)
            chart_cropped = chart[y:y+h, x:x+w]

            # Preserve the aspect ratio when resizing
            chart_resized = cv2.resize(chart_cropped, (w, h))  # Removed common height constraint

            chart_images.append(chart_resized)

    # If needed, adjust the concatenation process, for now, we return individual charts
    return chart_images

def group_text_by_font(text_data):
    grouped_text = []
    current_font_size = None
    current_block = []

    for text, size in text_data:
        if current_font_size is None:
            current_font_size = size

        if size == current_font_size:
            current_block.append(text)
        else:
            grouped_text.append({
                "text": " ".join(current_block),
                "font_size": current_font_size
            })
            current_block = [text]
            current_font_size = size

    # Append the last block of text
    if current_block:
        grouped_text.append({
            "text": " ".join(current_block),
            "font_size": current_font_size
        })

    return grouped_text

def verify_links(references):
    results = []
    for ref in references:
        if not ref.startswith(('http://', 'https://')):
            ref = 'http://' + ref
        try:
            response = requests.head(ref, allow_redirects=True, timeout=5)
            if response.status_code == 200:
                results.append((ref, 'Verified'))
            else:
                results.append((ref, 'Not Verified'))
        except requests.RequestException:
            results.append((ref, 'Not Verified'))
    return results

def is_valid_file_extension(filename):
    return filename.lower().endswith(tuple(ALLOWED_EXTENSIONS))

def compute_contrast(img):
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the mean and standard deviation of the image
    mean, stddev = cv2.meanStdDev(gray_image)

    # The cv2.meanStdDev function returns mean and stddev as single-element arrays, so we extract the values
    mean = mean[0][0]
    stddev = stddev[0][0]

    # Calculate contrast as the ratio of standard deviation to mean
    contrast = stddev / mean if mean else 0

    return contrast

from fuzzywuzzy import fuzz

def closest_font(text_from_image):
    img = Image.new('RGB', (200, 60), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    best_match = None
    highest_ratio = 0
    for font_name in KNOWN_FONTS:
        try:
            font = ImageFont.truetype(font_name + ".ttf", size=12)
            d.text((10, 10), text_from_image, font=font, fill=(0, 0, 0))
            rendered_text = pytesseract.image_to_string(img)
            match_ratio = fuzz.ratio(text_from_image, rendered_text)
            if match_ratio > highest_ratio:
                highest_ratio = match_ratio
                best_match = font_name
        except Exception as e:
            print(f"Error processing font {font_name}: {e}")
    return best_match

tool = language_tool_python.LanguageTool('en-US')
educational_keywords = [ "education", "learn", "teach", "knowledge", "school", "class", "lesson", "study", "student", "teacher", "curriculum", "academic", "training", "tutor", "instructor", "lecture", "course", "university", "college", "library", "textbook", "e-learning", "online learning", "study materials", "research", "homework", "assignment", "exam", "degree", "certificate", "diploma", "graduation", "scholarship", "academic institution", "pedagogy", "educational program", "academic achievement", "educational technology", "learning environment", "educational resources", "teaching method", "academic discipline", "academic department", "schooling", "educational system", "syllabus", "classroom", "study group", "educational materials", "learning objectives", "educational goals", "educational assessment", "educational psychology", "educational theory", "educational research", "educational development", "educational practice", "educational innovation", "educational philosophy", "academic literature", "educational workshop", "educational conference", "educational seminar", "educational event", "educational resource center", "educational project", "educational initiative", "educational funding", "educational grant", "educational association", "educational research institute", "educational consultant","educational approach","teaching philosophy","teaching strategy","learning strategy", ]

def categorize_image_content(img):
    try:

        # Perform OCR on the image to extract text
        extracted_text = pytesseract.image_to_string(img)

        # Check if any educational keywords are present in the extracted text
        for keyword in educational_keywords:
            if keyword.lower() in extracted_text.lower():
                return "Educational"

        # If no educational keywords are found
        return "Not Educational"

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and is_valid_file_extension(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = cv2.imread(filepath)
            category = categorize_image_content(img)
            if img is None:
                return "Error loading image", 500

            contrast = compute_contrast(img)

            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Calculate the average hue and saturation of the image
            # The Hue channel is in the range [0, 179] for OpenCV
            # The Saturation and Value channel are in the range [0, 255] for OpenCV
            hue_avg = (np.mean(hsv_image[:, :, 0]) * 2)/360  # scaling factor to convert to [0, 360] range
            saturation_avg = np.mean(hsv_image[:, :, 1]) / 255  # scaling to [0, 1] range


            custom_config = r'--oem 3 --psm 11'
            text_from_image = pytesseract.image_to_string(img, config=custom_config)

            matches = tool.check(text_from_image)
            error_details = []
            for match in matches:
                from_line = match.context[0:match.offset].count('\n') + 1
                from_column = match.offset - match.context.rfind('\n', 0, match.offset)
                error_detail = {
                    'from_line': from_line,
                    'from_column': from_column,
                    'message': match.message,
                    'replacements': match.replacements,
                    'length': match.errorLength
                }
                error_details.append(error_detail)

            words = re.findall(r'\b\w+\b', text_from_image.lower())
            images_found = find_charts(img)
            chart_images = find_charts(img)

            chart_images = find_charts(img)

            # Convert chart images to Base64 format
            base64_chart_images = []
            for chart in chart_images:
                _, buffer = cv2.imencode('.png', chart)
                base64_data = base64.b64encode(buffer).decode()
                base64_chart_images.append(base64_data)

            references = extract_references(text_from_image)
            verified_references = verify_links(references)

            d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            sentences = [text for text, conf in zip(d['text'], d['conf']) if float(conf) > 60.0]

            font_sizes = [(height / img.shape[0]) * 72 for height in d['height'] if height != 0]
            text_data = list(zip(sentences, font_sizes))

            detected_font = closest_font(text_from_image)
            paragraph_sizes = {}
            current_paragraph = ""
            paragraph = ""
            total_font_sizes = []
            for text, size in text_data:
                if text.strip():  # Check if the text is not empty
                    current_paragraph += text + " "
                    paragraph += text + " " 
                    total_font_sizes.append(size)
                else:
                    if current_paragraph.strip():  # Check if the paragraph is not empty
                        # Calculate the average font size for the current paragraph
                        average_font_size = sum(total_font_sizes) / len(total_font_sizes)
                        paragraph_sizes[current_paragraph.strip()] = average_font_size
                    current_paragraph = ""
                    total_font_sizes = []

            

            grouped_text = group_text_by_font(text_data)

            summary = summarize_text(text_from_image)

            return render_template(
                'results.html', 
                category = category,
                text_data = text_data,
                paragraph=paragraph, 
                hue_avg=hue_avg, 
                saturation_avg=saturation_avg, 
                contrast=contrast,
                file_type=file.content_type,
                grammar_errors=matches, 
                detected_font=detected_font, 
                references=verified_references,
                error_details=error_details,
                paragraph_sizes = paragraph_sizes,
                chart_images = base64_chart_images,
                chart_count=len(chart_images),
                grouped_text=grouped_text,
                text_from_image = text_from_image,
                summary = summary,
                
            )
    return render_template('index.html')

@app.route('/suggestions')
def suggestions():
    suggestions_data = request.args.get('suggestions', [])  # Adjust as needed for your implementation
    return render_template('suggestions.html', suggestions=suggestions_data)

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
