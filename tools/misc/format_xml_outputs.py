#!/usr/bin/env python3
"""
Pretty-format the XML outputs from the MJCF generator test for better readability.
"""

import xml.etree.ElementTree as ET
import os


def prettify_xml(xml_string):
    """Add proper indentation to XML string."""
    import xml.dom.minidom
    dom = xml.dom.minidom.parseString(xml_string)
    return dom.toprettyxml(indent="  ")


def format_all_outputs():
    """Format all XML outputs to be more readable."""
    input_dir = "/Users/hassanshahristani/Documents/IIB/4th_year_project/tools/test_outputs"
    output_dir = "/Users/hassanshahristani/Documents/IIB/4th_year_project/tools/test_outputs_formatted"
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.xml'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            print(f"Formatting {filename}...")
            
            with open(input_path, 'r') as f:
                content = f.read()
            
            # Extract just the XML content (remove the XML declaration line)
            xml_content = content.split('\n', 1)[1] if content.startswith('<?xml') else content
            
            try:
                formatted = prettify_xml(xml_content)
                
                with open(output_path, 'w') as f:
                    f.write(formatted)
                
                print(f"Saved formatted version to {output_path}")
                
                # Also print the content for immediate viewing
                print("="*60)
                print(f"FORMATTED {filename.upper()}:")
                print("="*60)
                print(formatted)
                print()
                
            except Exception as e:
                print(f"Error formatting {filename}: {e}")


if __name__ == "__main__":
    format_all_outputs()