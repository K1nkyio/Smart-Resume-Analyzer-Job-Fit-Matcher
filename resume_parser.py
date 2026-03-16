# resume_parser.py
import PyPDF2
import re
import docx
from io import BytesIO
import sys
from typing import Any, Dict, Tuple
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from datetime import datetime
import phonenumbers
from email_validator import validate_email, EmailNotValidError

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

try:
    from PIL import Image
except Exception:
    Image = None

def _safe_word_tokenize(text):
    """Tokenize words with a regex fallback if NLTK data is unavailable."""
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"\b\w+\b", text)

def _safe_sent_tokenize(text):
    """Tokenize sentences with a regex fallback if NLTK data is unavailable."""
    try:
        return sent_tokenize(text)
    except LookupError:
        return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]

if sys.version_info >= (3, 14):
    spacy = None
    print("Warning: spaCy is disabled on Python 3.14+ due dependency incompatibility.")
else:
    try:
        import spacy
    except Exception as e:
        spacy = None
        print(f"Warning: spaCy import failed ({e}). Advanced NLP matching will be disabled.")

# Download required NLTK data (run once)
required_nltk_resources = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
    ("corpora/words", "words"),
]
for resource_path, package_name in required_nltk_resources:
    try:
        nltk.data.find(resource_path)
    except Exception:
        try:
            nltk.download(package_name, quiet=True, force=True)
        except Exception:
            pass

# Load spaCy model for advanced NLP (install with: python -m spacy download en_core_web_sm)
if spacy is not None:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Warning: spaCy model unavailable ({e}). Install with: python -m spacy download en_core_web_sm")
        nlp = None
else:
    nlp = None


def _read_uploaded_bytes(file) -> bytes:
    """Read uploaded file bytes and reset pointer if possible."""
    if file is None:
        return b""

    if hasattr(file, "getvalue"):
        data = file.getvalue()
    else:
        data = file.read()

    try:
        file.seek(0)
    except Exception:
        pass

    return data or b""


def _ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    """OCR fallback for image-based PDFs. Returns empty string if OCR stack is unavailable."""
    if not pdf_bytes or convert_from_bytes is None or pytesseract is None:
        return ""

    try:
        images = convert_from_bytes(pdf_bytes, dpi=220)
    except Exception:
        return ""

    text_chunks = []
    for image in images:
        try:
            extracted = pytesseract.image_to_string(image)
            if extracted:
                text_chunks.append(extracted.strip())
        except Exception:
            continue

    return "\n".join(chunk for chunk in text_chunks if chunk)


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """OCR extraction for uploaded image resumes."""
    if not image_bytes or pytesseract is None or Image is None:
        return ""

    try:
        image = Image.open(BytesIO(image_bytes))
        return (pytesseract.image_to_string(image) or "").strip()
    except Exception:
        return ""

def extract_text_from_pdf(file, enable_ocr=True, return_ocr_used=False):
    """Extract text from PDF. Uses OCR fallback for image-based PDFs when available."""
    pdf_bytes = _read_uploaded_bytes(file)
    if not pdf_bytes:
        return ""

    extracted_pages = []
    empty_pages = 0

    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    extracted_pages.append(page_text)
                else:
                    empty_pages += 1
            except Exception as e:
                print(f"Error extracting text from page {page_num + 1}: {str(e)}")
                empty_pages += 1
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")

    extracted_text = "\n".join(extracted_pages).strip()
    needs_ocr = enable_ocr and (not extracted_text or empty_pages > 0)

    ocr_used = False
    if needs_ocr:
        ocr_text = _ocr_pdf_bytes(pdf_bytes)
        if ocr_text:
            ocr_used = True
            if extracted_text:
                extracted_text = f"{extracted_text}\n{ocr_text}"
            else:
                extracted_text = ocr_text

    final_text = extracted_text.strip()
    if return_ocr_used:
        return final_text, ocr_used
    return final_text

def extract_text_from_docx(file):
    """Extract text from DOCX files."""
    try:
        docx_bytes = _read_uploaded_bytes(file)
        doc = docx.Document(BytesIO(docx_bytes))
        text = ""
        
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + " "
                text += "\n"
        
        return text.strip()
    
    except Exception as e:
        print(f"Error reading DOCX file: {str(e)}")
        return ""

def extract_text_from_file(file, enable_ocr=True):
    """Extract text from PDF, DOCX, TXT, and image files."""
    file_type = (file.type if hasattr(file, "type") else file.name.split(".")[-1].lower()) or ""
    file_type = file_type.lower()

    if "pdf" in file_type:
        return extract_text_from_pdf(file, enable_ocr=enable_ocr)
    if "docx" in file_type or "document" in file_type:
        return extract_text_from_docx(file)
    if "txt" in file_type or "text" in file_type:
        raw = _read_uploaded_bytes(file)
        return raw.decode("utf-8", errors="ignore")
    if any(ext in file_type for ext in ["png", "jpg", "jpeg", "bmp", "tiff", "image"]):
        return _ocr_image_bytes(_read_uploaded_bytes(file)) if enable_ocr else ""

    raise ValueError(f"Unsupported file type: {file_type}")


def extract_text_with_metadata(file, enable_ocr=True) -> Dict[str, Any]:
    """Extract text and return parser metadata for UI explainability."""
    file_name = getattr(file, "name", "uploaded_file")
    file_type = (file.type if hasattr(file, "type") else file_name.split(".")[-1].lower()) or ""
    parser = "unknown"
    ocr_used = False
    lower_type = file_type.lower()
    if "pdf" in lower_type:
        parser = "pdf_parser"
        text, ocr_used = extract_text_from_pdf(file, enable_ocr=enable_ocr, return_ocr_used=True)
    elif "docx" in lower_type or "document" in lower_type:
        parser = "docx_parser"
        text = extract_text_from_docx(file)
    elif "txt" in lower_type or "text" in lower_type:
        parser = "text_parser"
        text = _read_uploaded_bytes(file).decode("utf-8", errors="ignore")
    elif any(ext in lower_type for ext in ["png", "jpg", "jpeg", "bmp", "tiff", "image"]):
        parser = "image_ocr_parser"
        ocr_used = True
        text = _ocr_image_bytes(_read_uploaded_bytes(file)) if enable_ocr else ""
    else:
        text = extract_text_from_file(file, enable_ocr=enable_ocr)

    return {
        "file_name": file_name,
        "file_type": file_type,
        "text": text,
        "parser": parser,
        "ocr_used": bool(ocr_used),
        "char_count": len(text or ""),
    }

def clean_text(text):
    """Enhanced text cleaning with options."""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s\-\.\@\(\)\+]', ' ', text)
    
    # Normalize common variations
    text = re.sub(r'\b(javascript|js)\b', 'JavaScript', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(c\+\+|cpp)\b', 'C++', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(c#|csharp)\b', 'C#', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(node\.?js|nodejs)\b', 'Node.js', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(react\.?js|reactjs)\b', 'React', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(vue\.?js|vuejs)\b', 'Vue.js', text, flags=re.IGNORECASE)
    
    return text

def extract_contact_info(text):
    """Extract contact information from resume text."""
    contact_info = {}
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        # Validate and clean emails
        valid_emails = []
        for email in emails:
            try:
                validated = validate_email(email)
                valid_emails.append(validated.email)
            except EmailNotValidError:
                continue
        contact_info['email'] = valid_emails[0] if valid_emails else None
    
    # Extract phone numbers
    phone_patterns = [
        r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        r'\(\d{3}\)\s*\d{3}[-.]?\d{4}'
    ]
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            # Format phone number consistently
            if isinstance(phones[0], tuple):
                contact_info['phone'] = f"({phones[0][0]}) {phones[0][1]}-{phones[0][2]}"
            else:
                contact_info['phone'] = phones[0]
            break
    
    # Extract LinkedIn profile
    linkedin_pattern = r'linkedin\.com/in/[\w\-]+'
    linkedin = re.search(linkedin_pattern, text, re.IGNORECASE)
    if linkedin:
        contact_info['linkedin'] = "https://" + linkedin.group()
    
    # Extract GitHub profile
    github_pattern = r'github\.com/[\w\-]+'
    github = re.search(github_pattern, text, re.IGNORECASE)
    if github:
        contact_info['github'] = "https://" + github.group()
    
    # Extract portfolio/website
    website_pattern = r'https?://(?:www\.)?[\w\-]+\.[\w\-]+(?:/[\w\-]*)*'
    websites = re.findall(website_pattern, text)
    if websites:
        # Filter out common sites that aren't personal websites
        personal_sites = [site for site in websites 
                         if not any(common in site.lower() 
                                  for common in ['linkedin', 'github', 'facebook', 'twitter'])]
        if personal_sites:
            contact_info['website'] = personal_sites[0]
    
    # Extract location (city, state)
    location_patterns = [
        r'([A-Z][a-z]+,\s*[A-Z]{2})',  # City, State
        r'([A-Z][a-z]+\s[A-Z][a-z]+,\s*[A-Z]{2})',  # City Name, State
        r'([A-Z][a-z]+,\s*[A-Z][a-z]+)'  # City, Country
    ]
    
    for pattern in location_patterns:
        location = re.search(pattern, text)
        if location:
            contact_info['location'] = location.group(1)
            break
    
    return contact_info

def extract_sections(text):
    """Extract and identify different sections of the resume."""
    sections = {}
    
    # Common section headers
    section_patterns = {
        'summary': r'(?:professional\s+)?(?:summary|profile|objective|about)',
        'experience': r'(?:professional\s+)?(?:experience|employment|work\s+history)',
        'education': r'education|academic\s+background|qualifications',
        'skills': r'(?:technical\s+)?skills|competencies|expertise',
        'projects': r'projects|portfolio|work\s+samples',
        'certifications': r'certifications|licenses|credentials',
        'awards': r'awards|achievements|honors|recognition',
        'languages': r'languages|linguistic\s+skills',
        'references': r'references|recommendations',
        'publications': r'publications|papers|articles'
    }
    
    text_lower = text.lower()
    lines = text.split('\n')
    
    for section_name, pattern in section_patterns.items():
        # Look for section headers
        for i, line in enumerate(lines):
            if re.search(pattern, line.lower().strip()) and len(line.strip()) < 50:
                # Found potential section header
                section_content = []
                j = i + 1
                
                # Extract content until next section or end
                while j < len(lines) and not any(re.search(p, lines[j].lower().strip()) 
                                               for p in section_patterns.values()):
                    if lines[j].strip():
                        section_content.append(lines[j].strip())
                    j += 1
                
                if section_content:
                    sections[section_name] = '\n'.join(section_content)
                break
    
    return sections

def extract_skills_advanced(text, skills_database):
    """Advanced skill extraction using NLP techniques."""
    found_skills = {}
    text_lower = text.lower()
    
    # Create a comprehensive list of all skills
    all_skills = []
    for category, skills in skills_database.items():
        all_skills.extend(skills.keys())
    
    # Method 1: Exact matching
    for category, skills in skills_database.items():
        found_skills[category] = []
        for skill, weight in skills.items():
            # Check for exact matches (case-insensitive)
            skill_lower = skill.lower()
            if skill_lower in text_lower:
                # Verify it's a whole word/phrase match
                pattern = r'\b' + re.escape(skill_lower) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills[category].append({
                        'skill': skill,
                        'weight': weight,
                        'confidence': 'high'
                    })
    
    # Method 2: Using spaCy for better context understanding
    if nlp:
        doc = nlp(text)
        
        # Extract noun phrases and entities
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        entities = [ent.text.lower() for ent in doc.ents]
        
        for category, skills in skills_database.items():
            for skill, weight in skills.items():
                skill_lower = skill.lower()
                
                # Check in noun phrases and entities
                if any(skill_lower in phrase for phrase in noun_phrases + entities):
                    if not any(s['skill'] == skill for s in found_skills[category]):
                        found_skills[category].append({
                            'skill': skill,
                            'weight': weight,
                            'confidence': 'medium'
                        })
    
    # Clean up empty categories
    found_skills = {k: v for k, v in found_skills.items() if v}
    
    return found_skills

def extract_experience_details(text):
    """Extract work experience details including dates and companies."""
    experiences = []
    
    # Patterns for dates
    date_patterns = [
        r'(\d{4})\s*[-–—]\s*(\d{4}|\w+)',  # 2020 - 2023
        r'(\w+\s+\d{4})\s*[-–—]\s*(\w+\s+\d{4}|\w+)',  # Jan 2020 - Dec 2023
        r'(\d{1,2}/\d{4})\s*[-–—]\s*(\d{1,2}/\d{4}|\w+)',  # 01/2020 - 12/2023
    ]
    
    # Pattern for job titles and companies
    title_company_pattern = r'([A-Z][a-zA-Z\s]+(?:Engineer|Manager|Director|Analyst|Developer|Designer|Specialist|Coordinator|Assistant|Lead|Senior|Junior))'
    
    lines = text.split('\n')
    current_experience = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for dates
        for pattern in date_patterns:
            date_match = re.search(pattern, line)
            if date_match:
                if current_experience:
                    experiences.append(current_experience)
                    current_experience = {}
                current_experience['dates'] = date_match.group()
                current_experience['raw_line'] = line
                break
        
        # Check for job titles
        title_match = re.search(title_company_pattern, line)
        if title_match and 'title' not in current_experience:
            current_experience['title'] = title_match.group(1).strip()
    
    if current_experience:
        experiences.append(current_experience)
    
    return experiences

def extract_education_details(text):
    """Extract education information including degrees, institutions, and years."""
    education = []
    
    # Degree patterns
    degree_patterns = [
        r'\b(PhD|Ph\.D\.?|Doctor of Philosophy)',
        r'\b(Master|M\.?S\.?|M\.?A\.?|MBA|M\.?Eng\.?)',
        r'\b(Bachelor|B\.?S\.?|B\.?A\.?|B\.?Eng\.?)',
        r'\b(Associate|A\.?S\.?|A\.?A\.?)',
        r'\b(Certificate|Diploma)'
    ]
    
    # Institution indicators
    institution_keywords = [
        'university', 'college', 'institute', 'school', 'academy'
    ]
    
    lines = text.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        
        # Check for degree
        for pattern in degree_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                # Check if line also contains institution keywords
                if any(keyword in line_lower for keyword in institution_keywords):
                    education.append({
                        'raw_text': line.strip(),
                        'type': 'formal_education'
                    })
                    break
    
    return education

def calculate_resume_completeness(sections, contact_info):
    """Calculate how complete the resume is based on standard sections."""
    required_sections = ['experience', 'education', 'skills']
    recommended_sections = ['summary', 'projects', 'certifications']
    contact_fields = ['email', 'phone']
    
    score = 0
    max_score = 100
    
    # Required sections (60% of score)
    for section in required_sections:
        if section in sections:
            score += 20
    
    # Recommended sections (25% of score)
    for section in recommended_sections:
        if section in sections:
            score += 8.33
    
    # Contact information (15% of score)
    for field in contact_fields:
        if contact_info.get(field):
            score += 7.5
    
    return min(score, max_score)

def analyze_resume_structure(text):
    """Analyze the overall structure and formatting of the resume."""
    analysis = {
        'word_count': len(text.split()),
        'character_count': len(text),
        'line_count': len(text.split('\n')),
        'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
        'has_bullet_points': bool(re.search(r'[•·▪▫‣⁃]', text)),
        'has_numbers': bool(re.search(r'\d+', text)),
        'has_dates': bool(re.search(r'\b\d{4}\b', text)),
        'has_email': bool(re.search(r'@', text)),
        'has_phone': bool(re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text))
    }
    
    # Readability score (simplified)
    sentences = _safe_sent_tokenize(text)
    if sentences:
        avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences)
        analysis['avg_sentence_length'] = round(avg_sentence_length, 2)
    
    # Estimate resume length in pages (assuming ~500 words per page)
    analysis['estimated_pages'] = max(1, round(analysis['word_count'] / 500))
    
    return analysis

def extract_keywords_from_job_description(job_desc):
    """Extract important keywords from job description for ATS optimization."""
    if not job_desc:
        return []
    
    # Clean and tokenize
    clean_desc = clean_text(job_desc)
    words = _safe_word_tokenize(clean_desc.lower())
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        stop_words = set()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Get word frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, freq in sorted_words[:20]]

def generate_resume_suggestions(sections, contact_info, structure_analysis):
    """Generate specific suggestions for resume improvement."""
    suggestions = []
    
    # Contact information suggestions
    if not contact_info.get('email'):
        suggestions.append("Add a professional email address")
    if not contact_info.get('phone'):
        suggestions.append("Include a phone number")
    if not contact_info.get('linkedin'):
        suggestions.append("Add your LinkedIn profile URL")
    
    # Section suggestions
    if 'summary' not in sections:
        suggestions.append("Add a professional summary at the top")
    if 'projects' not in sections:
        suggestions.append("Include a projects section to showcase your work")
    if 'certifications' not in sections:
        suggestions.append("Add relevant certifications if you have any")
    
    #
