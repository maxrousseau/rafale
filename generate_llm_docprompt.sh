#!/bin/bash

# Output file
OUTPUT_FILE="llm-docprompt.txt"

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Add repo structure, excluding unwanted directories like .venv
echo "### Repository Structure ###" >> "$OUTPUT_FILE"
tree . -I 'wandb|*__pycache__|media|*-requirements.txt|.venv' >> "$OUTPUT_FILE"
echo -e "\n\n" >> "$OUTPUT_FILE"

# Include README.md content
if [[ -f "README.md" ]]; then
    echo "### README.md ###" >> "$OUTPUT_FILE"
    cat README.md >> "$OUTPUT_FILE"
    echo -e "\n\n" >> "$OUTPUT_FILE"
fi

# Function to include content of a given file type, excluding .venv directory
include_files() {
    local pattern="$1"
    local header="$2"

    find . -type f -name "$pattern" ! -path "./.venv/*" | while read -r file; do
        echo "### $file ###" >> "$OUTPUT_FILE"
        cat "$file" >> "$OUTPUT_FILE"
        echo -e "\n\n" >> "$OUTPUT_FILE"
    done
}

# Include Python files, excluding those in .venv
include_files "*.py" "Python File"

# Include YAML files only from the 'test' folder
find ./test -type f -name "*.yaml" | while read -r yaml_file; do
    echo "### $yaml_file ###" >> "$OUTPUT_FILE"
    cat "$yaml_file" >> "$OUTPUT_FILE"
    echo -e "\n\n" >> "$OUTPUT_FILE"
done

echo "Documentation prompt has been generated in $OUTPUT_FILE"
