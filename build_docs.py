#!/usr/bin/env python3
"""
Build documentation from Markdown to PDF using md-to-pdf.

This script provides a build hook to convert all markdown documentation
files to PDF format using the md-to-pdf npm package (which uses Puppeteer).
"""

import os
import subprocess
import sys
from pathlib import Path
import json
from typing import List, Dict, Optional
import concurrent.futures
import time
import shutil


class DocumentationBuilder:
    """Handles the conversion of Markdown documentation to PDF."""
    
    def __init__(self, source_dir: str = "src/smartsurge/docs", output_dir: str = "dist/docs/pdf"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.markdown_files: List[Path] = []
        
    def check_dependencies(self) -> bool:
        """Check if md-to-pdf is installed."""
        # Check if node_modules exists
        node_modules = Path("node_modules/.bin/md-to-pdf")
        if node_modules.exists():
            print("✓ md-to-pdf is installed locally")
            return True
            
        print("✗ md-to-pdf is not installed")
        print("Installing dependencies...")
        
        try:
            subprocess.run(["npm", "install"], check=True)
            print("✓ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install dependencies: {e}")
            return False
            
    def collect_markdown_files(self) -> None:
        """Recursively collect all markdown files from the source directory."""
        self.markdown_files = list(self.source_dir.rglob("*.md"))
        print(f"Found {len(self.markdown_files)} markdown files")
        
    def create_output_structure(self) -> None:
        """Create the output directory structure matching the source."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for md_file in self.markdown_files:
            relative_path = md_file.relative_to(self.source_dir)
            output_path = self.output_dir / relative_path.with_suffix(".pdf")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
    def generate_pdf_config(self) -> Dict:
        """Generate configuration for md-to-pdf."""
        return {
            "pdf_options": {
                "format": "A4",
                "margin": {
                    "top": "20mm",
                    "right": "20mm",
                    "bottom": "20mm",
                    "left": "20mm"
                },
                "printBackground": True,
                "displayHeaderFooter": True,
                "headerTemplate": """<div style="font-size: 10px; width: 100%; text-align: center; color: #666;">SmartSurge Documentation</div>""",
                "footerTemplate": """<div style="font-size: 10px; width: 100%; text-align: center; color: #666;">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>"""
            },
            "stylesheet": [
                # Inline CSS as a data URL
                "data:text/css;base64," + self.get_encoded_css()
            ],
            "highlight_style": "github",
            "marked_options": {
                "breaks": False,
                "gfm": True
            }
        }
        
    def get_encoded_css(self) -> str:
        """Get base64 encoded CSS."""
        import base64
        css = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 12pt;
    line-height: 1.6;
    color: #333;
}

h1, h2, h3, h4, h5, h6 {
    margin-top: 24px;
    margin-bottom: 16px;
    font-weight: 600;
}

h1 {
    font-size: 24pt;
    border-bottom: 2px solid #e1e4e8;
    padding-bottom: 8px;
    page-break-before: always;
}

h1:first-of-type {
    page-break-before: auto;
}

h2 {
    font-size: 20pt;
    border-bottom: 1px solid #e1e4e8;
    padding-bottom: 6px;
}

h3 {
    font-size: 16pt;
}

code {
    background-color: #f6f8fa;
    padding: 2px 4px;
    border-radius: 3px;
    font-family: Consolas, Monaco, 'Courier New', monospace;
    font-size: 0.9em;
}

pre {
    background-color: #f6f8fa;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
    line-height: 1.4;
}

pre code {
    background-color: transparent;
    padding: 0;
    font-size: 0.85em;
}

blockquote {
    border-left: 4px solid #dfe2e5;
    margin: 0;
    padding-left: 16px;
    color: #666;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
}

table th, table td {
    border: 1px solid #dfe2e5;
    padding: 8px 12px;
}

table th {
    background-color: #f6f8fa;
    font-weight: 600;
}

a {
    color: #0366d6;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}
        """
        return base64.b64encode(css.encode()).decode()
        
    def convert_file(self, md_file: Path) -> bool:
        """Convert a single markdown file to PDF."""
        relative_path = md_file.relative_to(self.source_dir)
        output_file = self.output_dir / relative_path.with_suffix(".pdf")
        
        print(f"Converting: {relative_path}")
        
        # Create a temporary config file
        config = self.generate_pdf_config()
        config_file = Path(f".md-to-pdf-config-{md_file.stem}.json")
        
        try:
            # Write config
            config_file.write_text(json.dumps(config, indent=2))
            
            # Run md-to-pdf
            cmd = [
                "node_modules/.bin/md-to-pdf",
                str(md_file),
                "--config-file", str(config_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Move the generated PDF to the output directory
            generated_pdf = md_file.with_suffix(".pdf")
            if generated_pdf.exists():
                shutil.move(str(generated_pdf), str(output_file))
                print(f"  ✓ Generated: {output_file.relative_to(self.output_dir.parent.parent)}")
                return True
            else:
                print(f"  ✗ PDF not generated for {md_file}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error converting {md_file.name}: {e}")
            if e.stderr:
                print(f"    {e.stderr[:200]}...")  # Truncate long errors
            return False
        finally:
            # Clean up config file
            if config_file.exists():
                config_file.unlink()
            
    def convert_files_sequential(self) -> int:
        """Convert files sequentially to avoid file conflicts."""
        success_count = 0
        
        for md_file in self.markdown_files:
            if self.convert_file(md_file):
                success_count += 1
                
        return success_count
        
    def generate_index(self) -> None:
        """Generate an index HTML file listing all PDFs."""
        index_content = """<!DOCTYPE html>
<html>
<head>
    <title>SmartSurge Documentation PDFs</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 { 
            color: #333;
            margin-bottom: 30px;
        }
        .category {
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 18px;
            font-weight: bold;
            color: #555;
            border-bottom: 2px solid #e1e4e8;
            padding-bottom: 8px;
        }
        ul { 
            list-style-type: none; 
            padding: 0;
            margin: 0;
        }
        li { 
            margin: 8px 0;
        }
        a {
            color: #0366d6;
            text-decoration: none;
            padding: 8px 12px;
            display: inline-block;
            border-radius: 4px;
            transition: background-color 0.2s;
        }
        a:hover {
            background-color: #f6f8fa;
        }
        .pdf-icon {
            display: inline-block;
            width: 16px;
            height: 16px;
            margin-right: 8px;
            vertical-align: middle;
        }
        .info {
            margin-top: 30px;
            padding: 15px;
            background-color: #f6f8fa;
            border-radius: 4px;
            font-size: 14px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 SmartSurge Documentation PDFs</h1>
"""
        
        # Group PDFs by category
        pdfs = list(self.output_dir.rglob("*.pdf"))
        categories = {}
        
        for pdf in sorted(pdfs):
            relative = pdf.relative_to(self.output_dir)
            category = relative.parts[0] if len(relative.parts) > 1 else "Main"
            if category not in categories:
                categories[category] = []
            categories[category].append(relative)
            
        # Add PDF count
        index_content += f'        <p class="info">Total PDFs generated: {len(pdfs)}</p>\n'
            
        for category, files in sorted(categories.items()):
            index_content += f'        <div class="category">{category.title()}</div>\n        <ul>\n'
            for file in sorted(files):
                title = file.stem.replace("_", " ").title()
                index_content += f'            <li><span class="pdf-icon">📄</span><a href="{file.as_posix()}">{title}</a></li>\n'
            index_content += '        </ul>\n'
            
        index_content += """        <div class="info">
            Generated with md-to-pdf (Puppeteer-based PDF generation)
        </div>
    </div>
</body>
</html>"""
        
        index_file = self.output_dir / "index.html"
        index_file.write_text(index_content)
        print(f"\n✓ Generated index: {index_file}")
        
    def build(self) -> bool:
        """Main build process."""
        print("SmartSurge Documentation PDF Builder")
        print("=" * 40)
        
        start_time = time.time()
        
        # Check dependencies
        if not self.check_dependencies():
            return False
            
        # Collect markdown files
        self.collect_markdown_files()
        if not self.markdown_files:
            print("No markdown files found!")
            return False
            
        # Create output structure
        self.create_output_structure()
        
        # Convert files sequentially (to avoid conflicts)
        print(f"\nConverting {len(self.markdown_files)} files...")
        success_count = self.convert_files_sequential()
        
        print(f"\n✓ Successfully converted {success_count}/{len(self.markdown_files)} files")
        
        # Generate index
        self.generate_index()
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Build complete in {elapsed_time:.1f} seconds!")
        print(f"✓ PDFs available in: {self.output_dir}")
        
        return success_count > 0


def main():
    """Main entry point with build hook support."""
    # Check if this is being called as a build hook
    if len(sys.argv) > 1 and sys.argv[1] == "--hook":
        print("Running as build hook...")
        
    builder = DocumentationBuilder()
    success = builder.build()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()