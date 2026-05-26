//for checking chunk size depending on the doc size

python -c "
from pypdf import PdfReader
reader = PdfReader('docs/document.pdf')
text = ''.join(p.extract_text() for p in reader.pages)
paragraphs = [p for p in text.split('\n\n') if len(p.strip()) > 50]
avg = sum(len(p) for p in paragraphs) // len(paragraphs)
print(f'Average paragraph length: {avg} characters')
print(f'Recommended chunk_size: {avg}')
"


//for checking chunk size depending on the doc size the best to check

python -c "
from pypdf import PdfReader
reader = PdfReader('docs/document.pdf')
text = ''.join(p.extract_text() for p in reader.pages)
lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 80]
avg = sum(len(l) for l in lines) // len(lines)
print(f'Average line length: {avg} characters')
print(f'Recommended chunk_size: {avg * 4}')
"
