#!/usr/bin/env python3
"""
Minimal setup.py for custom build commands.
All configuration is in pyproject.toml, this only handles custom build steps.
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.install import install
from setuptools.command.develop import develop


class BuildPDFDocs:
    """Mixin class for building PDF documentation."""
    
    def build_pdfs(self):
        """Build PDF documentation if installing with [docs] extra."""
        # Check if we're installing with the docs extra
        # This is a bit hacky but works for most cases
        installing_docs = any(
            'docs' in arg for arg in sys.argv 
            if '[' in arg and ']' in arg
        ) or os.environ.get('SMARTSURGE_BUILD_DOCS', '').lower() == 'true'
        
        if not installing_docs:
            return
            
        print("Building PDF documentation...")
        
        # Find build_docs.py
        possible_paths = [
            Path.cwd() / 'build_docs.py',
            Path(__file__).parent / 'build_docs.py',
            Path.cwd() / 'src' / 'build_docs.py',
        ]
        
        build_docs_path = None
        for path in possible_paths:
            if path.exists():
                build_docs_path = path
                break
                
        if not build_docs_path:
            print("Warning: build_docs.py not found, skipping PDF generation")
            return
            
        try:
            # Run the build hook
            result = subprocess.run(
                [sys.executable, str(build_docs_path), '--hook'],
                capture_output=True,
                text=True,
                cwd=build_docs_path.parent
            )
            
            if result.returncode == 0:
                print("PDF documentation built successfully")
            else:
                print(f"Warning: PDF build failed with code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                    
        except Exception as e:
            print(f"Warning: Failed to build PDFs: {e}")
            # Don't fail the installation if PDF building fails


class BuildPyWithPDFs(build_py, BuildPDFDocs):
    """Custom build_py command that also builds PDFs."""
    
    def run(self):
        # Build PDFs before building Python files
        self.build_pdfs()
        super().run()


class InstallWithPDFs(install, BuildPDFDocs):
    """Custom install command that also builds PDFs."""
    
    def run(self):
        # Build PDFs before installation
        self.build_pdfs()
        super().run()


class DevelopWithPDFs(develop, BuildPDFDocs):
    """Custom develop command that also builds PDFs."""
    
    def run(self):
        # Build PDFs before development installation
        self.build_pdfs()
        super().run()


# Minimal setup() call - all configuration is in pyproject.toml
setup(
    cmdclass={
        'build_py': BuildPyWithPDFs,
        'install': InstallWithPDFs,
        'develop': DevelopWithPDFs,
    }
)