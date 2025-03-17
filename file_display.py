from __future__ import annotations

import os
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from loguru import logger


class FileList(list):
    """A list subclass that can display as HTML in Jupyter notebooks."""
    
    def __init__(self, items, parent):
        super().__init__(items)
        self._parent = parent
    
    def _repr_html_(self):
        """Return HTML representation for Jupyter notebooks."""
        return self._parent._files_repr_html_()


def generate_files_html(files: List[str], datasite_email: str, 
                        instance_id: Optional[str] = None,
                        file_permissions: Optional[Dict[str, List[str]]] = None) -> str:
    """Generate HTML representation for a list of files with permission information.
    
    Args:
        files: List of file paths
        datasite_email: Email of the datasite owner
        instance_id: Optional unique ID for this instance (generated if None)
        file_permissions: Optional dictionary of file paths to lists of users with permissions
        
    Returns:
        HTML string with the file explorer widget
    """
    try:
        if not files:
            return "<div><em>No accessible files found</em></div>"
        
        # Generate a unique ID for this instance to avoid conflicts between multiple outputs
        if instance_id is None:
            instance_id = f"files_{hash(datasite_email) % 100000}_{int(time.time() * 1000) % 100000}"
        
        # Find common path prefixes to auto-expand
        common_prefixes = _find_common_prefixes(files)
        
        # Build a hierarchical tree structure from file paths
        file_tree = _build_file_tree(files, file_permissions)
        
        # Render the tree structure to HTML
        file_tree_html = _render_tree(file_tree, common_prefixes, instance_id)
        
        # Return the complete HTML output
        return _assemble_html(file_tree_html, files, datasite_email, instance_id)
        
    except Exception as e:
        # Fallback to a simple list if anything fails
        logger.error(f"Error creating HTML file representation: {e}")
        return f"<div><p>Files in {datasite_email} ({len(files)} files):</p><ul>" + \
               "".join([f"<li>{file}</li>" for file in files[:20]]) + \
               (f"<li>... and {len(files) - 20} more</li>" if len(files) > 20 else "") + \
               "</ul></div>"


def _find_common_prefixes(files: List[str]) -> List[str]:
    """Find common directory prefixes in a list of file paths."""
    common_prefixes = []
    if not files:
        return common_prefixes
        
    # Normalize paths with forward slashes
    normalized_paths = [p.replace('\\', '/') for p in files]
    
    # Find the common prefix of all paths
    if len(normalized_paths) > 1:
        # Get the first path as a starting point
        common_prefix = normalized_paths[0]
        
        # Check if this prefix is in all paths
        while common_prefix:
            if all(p.startswith(common_prefix + '/') for p in normalized_paths):
                common_prefixes.append(common_prefix)
                # Get the next level deeper
                subdirs = set()
                for path in normalized_paths:
                    # Get the next directory in the path after the common prefix
                    relative = path[len(common_prefix)+1:]
                    next_dir = relative.split('/', 1)[0]
                    if '/' in relative:  # Make sure there's another level
                        subdirs.add(next_dir)
            
                # If all files are in the same subdirectory, expand to that level too
                if len(subdirs) == 1:
                    common_prefix = common_prefix + '/' + next(iter(subdirs))
                else:
                    break
            else:
                break
    
    # Add the parent of each common prefix to ensure proper nesting
    for prefix in list(common_prefixes):
        parent = os.path.dirname(prefix)
        if parent and parent not in common_prefixes:
            common_prefixes.append(parent)
            
    return common_prefixes


def _build_file_tree(files: List[str], file_permissions: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Build a hierarchical tree structure from file paths, including permission information.
    
    Args:
        files: List of file paths
        file_permissions: Optional dictionary mapping file paths to lists of users with permissions
        
    Returns:
        Hierarchical dictionary representing the file tree
    """
    file_tree = {}
    for file_path in files:
        # Get just the filename for the leaf node
        filename = os.path.basename(file_path)
        
        # Get permissions for this file if available
        permissions = file_permissions.get(file_path, []) if file_permissions else []
        
        # Split the directory structure
        dir_path = os.path.dirname(file_path)
        if not dir_path:
            # File is at root level
            if '__files__' not in file_tree:
                file_tree['__files__'] = []
            file_tree['__files__'].append({
                'name': filename,
                'path': file_path,
                'permissions': permissions
            })
            continue
            
        # Normalize path separators
        dir_path = dir_path.replace('\\', '/').rstrip('/')
        path_parts = dir_path.split('/')
        
        # Build the directory structure
        current = file_tree
        for part in path_parts:
            if not part:  # Skip empty parts
                continue
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Add the file to the leaf directory
        if '__files__' not in current:
            current['__files__'] = []
        current['__files__'].append({
            'name': filename,
            'path': file_path,
            'permissions': permissions
        })
    
    return file_tree


def _render_tree(tree: Dict[str, Any], common_prefixes: List[str], instance_id: str, level: int = 0, current_path: str = "") -> str:
    """Render the file tree structure as HTML, including permission information.
    
    Args:
        tree: Hierarchical dictionary representing the file tree
        common_prefixes: List of directory prefixes to auto-expand
        instance_id: Unique identifier for this instance
        level: Current nesting level (used for ID generation)
        current_path: Current path in the tree (used for auto-expansion)
        
    Returns:
        HTML string representation of the tree
    """
    html = []
    
    # First, render files at this level
    if '__files__' in tree:
        for file in tree['__files__']:
            file_name = file['name']
            file_path = file['path']
            permissions = file.get('permissions', [])
            escaped_path = file_path.replace("'", "\\'").replace('"', '\\"')
            
            # Create a searchable permission string and escape it
            perm_search_data = " ".join(permissions)
            escaped_perm_data = perm_search_data.replace("'", "\\'").replace('"', '\\"')
            
            # Format permissions info for display
            perm_count = len(permissions)
            if "*" in permissions:
                perm_text = '<span class="perm-count">Everyone</span>'
            else:
                perm_text = f'<span class="perm-count">{perm_count} user{"s" if perm_count != 1 else ""}</span>'
                if perm_count > 0:
                    user_pills = ' '.join([f'<span class="user-pill">{user}</span>' for user in permissions])
                    perm_text += f' <span class="user-list">{user_pills}</span>'
            
            # Determine file type
            _, ext = os.path.splitext(file_name)
            file_type = ext[1:].lower() if ext else 'unknown'
            
            html.append(f"""
            <div class="file-item" data-path="{escaped_path}" data-permissions="{escaped_perm_data}">
                <input type="checkbox" class="file-checkbox" data-path="{escaped_path}" 
                       onchange="toggleSelection_{instance_id}(this, '{escaped_path}')">
                <span class="file-icon file-{file_type}"></span>
                <span class="file-name">{file_name}</span>
                <span class="file-permissions">{perm_text}</span>
            </div>
            """)
    
    # Then render subfolders
    for key in sorted(tree.keys()):
        if key != '__files__':
            subfolder = tree[key]
            folder_id = f"folder_{key}_{level}_{hash(key) % 10000}_{instance_id}"
            
            # Determine the full path of this folder
            folder_path = f"{current_path}/{key}".lstrip('/')
            
            # Check if this folder should be auto-expanded
            auto_expand = folder_path in common_prefixes
            folder_class = "folder open" if auto_expand else "folder"
            folder_icon = "▼" if auto_expand else "▶"
            folder_type = "📂" if auto_expand else "📁"
            
            html.append(f"""
            <div class="{folder_class}" id="{folder_id}">
                <div class="folder-item">
                    <input type="checkbox" class="file-checkbox" 
                           onchange="toggleFolderCheckbox_{instance_id}(this, document.getElementById('{folder_id}'))">
                    <span class="folder-toggle" onclick="toggleFolder_{instance_id}(this)">{folder_icon}</span>
                    <span class="folder-icon">{folder_type}</span>
                    <span class="folder-name">{key}</span>
                </div>
                <div class="folder-content">
                    {_render_tree(subfolder, common_prefixes, instance_id, level+1, folder_path)}
                </div>
            </div>
            """)
    
    return ''.join(html)


def _assemble_html(file_tree_html: str, files: List[str], datasite_email: str, instance_id: str) -> str:
    """Assemble the complete HTML output with CSS and JavaScript."""
    # CSS for the file explorer
    css_code = f"""
    <style>
    /* File container styles */
    #file-container-{instance_id} {{
        font-family: Arial, sans-serif;
        margin: 20px 0;
        max-width: 100%;
        border: 1px solid #ddd;
        border-radius: 8px;
        overflow: hidden;
    }}
    
    /* Header styles */
    #file-header-{instance_id} {{
        background-color: #f5f5f5;
        padding: 12px 15px;
        border-bottom: 1px solid #ddd;
    }}
    #file-header-{instance_id} h3 {{
        margin: 0;
        font-size: 16px;
        color: #333;
    }}
    
    /* Search and controls */
    #file-search-controls-{instance_id} {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 15px;
        background-color: #f9f9f9;
        border-bottom: 1px solid #ddd;
    }}
    #file-search-{instance_id} {{
        flex-grow: 1;
        margin-right: 15px;
    }}
    #file-search-input-{instance_id} {{
        padding: 8px;
        width: 100%;
        box-sizing: border-box;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 14px;
    }}
    #copy-controls-{instance_id} {{
        display: flex;
        align-items: center;
        white-space: nowrap;
    }}
    #copy-selected-btn-{instance_id} {{
        background-color: #f0f0f0;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 8px 12px;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.2s;
    }}
    #copy-selected-btn-{instance_id}:hover:not(:disabled) {{
        background-color: #e0e0e0;
    }}
    #copy-selected-btn-{instance_id}:disabled {{
        cursor: not-allowed;
        opacity: 0.6;
    }}
    #copy-selected-btn-{instance_id}.active {{
        background-color: #4285f4;
        color: white;
        border-color: #2a75f3;
    }}
    #copy-selected-btn-{instance_id}.copy-success {{
        background-color: #34a853;
        color: white;
        border-color: #2d9249;
    }}
    #copy-selected-btn-{instance_id}.copy-error {{
        background-color: #ea4335;
        color: white;
        border-color: #d33426;
    }}
    #selected-info-{instance_id} {{
        margin-right: 10px;
        font-size: 14px;
    }}
    
    /* File Explorer Styles */
    #file-explorer-{instance_id} {{
        background-color: #fff;
        font-size: 14px;
        max-height: 500px;
        overflow: auto;
        padding: 10px;
    }}
    .folder {{
        margin: 4px 0;
        padding-left: 0;
    }}
    .folder-content {{
        display: none; /* Start collapsed */
        padding-left: 22px;
        border-left: 1px solid #eee;
        margin-left: 5px;
    }}
    .folder.open > .folder-content {{
        display: block;
    }}
    .folder-item, .file-item {{
        display: flex;
        align-items: center;
        padding: 3px 0;
        margin: 2px 0;
    }}
    .folder-item:hover, .file-item:hover {{
        background-color: #f5f5f5;
        border-radius: 4px;
    }}
    .folder-label {{
        display: flex;
        align-items: center;
        flex: 1;
    }}
    .file-checkbox {{
        margin-right: 5px;
        cursor: pointer;
    }}
    .folder-toggle {{
        display: inline-block;
        width: 16px;
        height: 16px;
        text-align: center;
        line-height: 16px;
        cursor: pointer;
        color: #555;
        margin-right: 3px;
        font-size: 10px;
    }}
    .folder-icon {{
        margin-right: 5px;
        font-size: 1.1em;
    }}
    .file-icon {{
        margin-right: 5px;
        font-size: 1.1em;
        color: #888;
    }}
    .folder-name {{
        cursor: default;
        user-select: none;
    }}
    .file-name {{
        flex: 0 0 auto;
        margin-right: 10px;
    }}
    
    /* Enhanced permission styling */
    .file-permissions {{
        color: #666;
        font-size: 12px;
        margin-left: 10px;
        display: flex;
        align-items: center;
        flex-wrap: nowrap;
        overflow: hidden;
        max-width: 60%;
    }}
    .perm-count {{
        color: #555;
        margin-right: 5px;
    }}
    .user-list {{
        display: flex;
        flex-wrap: nowrap;
        align-items: center;
        overflow-x: auto;
        max-width: 100%;
        padding-bottom: 3px; /* Room for scrollbar */
    }}
    .user-pill {{
        display: inline-block;
        background-color: #f0f0f0;
        color: #666;
        border-radius: 12px;
        padding: 2px 8px;
        margin-right: 4px;
        white-space: nowrap;
        font-size: 11px;
        border: 1px solid #e0e0e0;
    }}
    
    /* File type icons */
    .file-py::before {{ content: '🐍'; }}
    .file-js::before {{ content: '📜'; }}
    .file-html::before, .file-htm::before {{ content: '🌐'; }}
    .file-css::before {{ content: '🎨'; }}
    .file-json::before {{ content: '📋'; }}
    .file-md::before {{ content: '📝'; }}
    .file-txt::before {{ content: '📄'; }}
    .file-pdf::before {{ content: '📕'; }}
    .file-jpg::before, .file-jpeg::before, .file-png::before, 
    .file-gif::before, .file-svg::before {{ content: '🖼️'; }}
    .file-mp3::before, .file-wav::before, .file-ogg::before {{ content: '🎵'; }}
    .file-mp4::before, .file-avi::before, .file-mov::before {{ content: '🎬'; }}
    .file-zip::before, .file-rar::before, .file-tar::before {{ content: '📦'; }}
    .file-unknown::before {{ content: '📄'; }}
    </style>
    """
    
    # JavaScript with updated search function
    js_code = f"""
    <script>
    // Use an IIFE to create a closure and avoid global variable conflicts
    (function() {{
        // Track selected files - use unique variable name with instance ID
        const selectedFiles_{instance_id} = new Set();
        
        // Define all functions within this scope
        window.toggleFolder_{instance_id} = function(element) {{
            const folder = element.closest('.folder');
            folder.classList.toggle('open');
            
            // Update the folder toggle icon
            if (folder.classList.contains('open')) {{
                element.textContent = '▼';
                folder.querySelector('.folder-icon').textContent = '📂';
            }} else {{
                element.textContent = '▶';
                folder.querySelector('.folder-icon').textContent = '📁';
            }}
        }};
        
        window.toggleSelection_{instance_id} = function(checkbox, filePath) {{
            if (checkbox.checked) {{
                selectedFiles_{instance_id}.add(filePath);
            }} else {{
                selectedFiles_{instance_id}.delete(filePath);
            }}
            updateCopyButton_{instance_id}();
            updateParentFolderCheckboxes_{instance_id}(checkbox);
        }};
        
        window.updateParentFolderCheckboxes_{instance_id} = function(changedCheckbox) {{
            const folderItem = changedCheckbox.closest('.folder-item, .file-item');
            const parentFolder = folderItem ? folderItem.closest('.folder') : null;
            if (!parentFolder) return;
            
            const parentCheckbox = parentFolder.querySelector('> .folder-item > .file-checkbox');
            if (!parentCheckbox) return;
            
            // Get all checkboxes in this folder
            const childCheckboxes = Array.from(
                parentFolder.querySelectorAll('.file-checkbox')
            ).filter(cb => cb !== parentCheckbox);
            
            // Count checked and total checkboxes
            const totalCheckboxes = childCheckboxes.length;
            const checkedCheckboxes = childCheckboxes.filter(cb => cb.checked).length;
            
            if (checkedCheckboxes === 0) {{
                // None checked
                parentCheckbox.checked = false;
                parentCheckbox.indeterminate = false;
            }} else if (checkedCheckboxes === totalCheckboxes) {{
                // All checked
                parentCheckbox.checked = true;
                parentCheckbox.indeterminate = false;
            }} else {{
                // Some checked
                parentCheckbox.checked = false;
                parentCheckbox.indeterminate = true;
            }}
            
            // Continue up the tree
            updateParentFolderCheckboxes_{instance_id}(parentCheckbox);
        }};
        
        window.toggleFolderCheckbox_{instance_id} = function(checkbox, folderElement) {{
            // Get all checkboxes within this folder (files and subfolders)
            const allCheckboxes = folderElement.querySelectorAll('.file-checkbox');
            
            // Set all to match the folder checkbox
            allCheckboxes.forEach(cb => {{
                cb.checked = checkbox.checked;
                
                // For file checkboxes, update the selectedFiles set
                if (cb.hasAttribute('data-path')) {{
                    const filePath = cb.getAttribute('data-path');
                    if (checkbox.checked) {{
                        selectedFiles_{instance_id}.add(filePath);
                    }} else {{
                        selectedFiles_{instance_id}.delete(filePath);
                    }}
                }}
            }});
            
            // Update the copy button
            updateCopyButton_{instance_id}();
            
            // Update parent folders
            updateParentFolderCheckboxes_{instance_id}(checkbox);
        }};
        
        window.updateCopyButton_{instance_id} = function() {{
            const btn = document.getElementById('copy-selected-btn-{instance_id}');
            const count = document.getElementById('selected-count-{instance_id}');
            
            if (!btn || !count) return;
            
            count.textContent = selectedFiles_{instance_id}.size;
            
            if (selectedFiles_{instance_id}.size > 0) {{
                btn.disabled = false;
                btn.classList.add('active');
            }} else {{
                btn.disabled = true;
                btn.classList.remove('active');
            }}
        }};
        
        window.copySelectedToPythonList_{instance_id} = function() {{
            // Format as a Python list
            const fileArray = Array.from(selectedFiles_{instance_id});
            const pythonListStr = "my_files = [" + 
                fileArray.map(path => "'" + path.replace(/'/g, "\\'") + "'").join(", ") + 
                "]";
            
            // Create a temporary textarea
            const textarea = document.createElement('textarea');
            textarea.value = pythonListStr;
            textarea.setAttribute('readonly', '');
            textarea.style.position = 'absolute';
            textarea.style.left = '-9999px';
            document.body.appendChild(textarea);
            
            // Select and copy
            textarea.select();
            try {{
                const successful = document.execCommand('copy');
                if (successful) {{
                    showCopiedTooltip_{instance_id}();
                }} else {{
                    showCopyError_{instance_id}();
                }}
            }} catch (err) {{
                showCopyError_{instance_id}();
                console.error('Failed to copy text: ', err);
            }}
            
            // Clean up
            document.body.removeChild(textarea);
        }};
        
        window.showCopiedTooltip_{instance_id} = function() {{
            const copyButton = document.getElementById('copy-selected-btn-{instance_id}');
            const originalText = copyButton.textContent;
            
            // Change button text to show success
            copyButton.textContent = '✓ Copied as Python List!';
            copyButton.classList.add('copy-success');
            
            // Reset after delay
            setTimeout(() => {{
                copyButton.textContent = originalText;
                copyButton.classList.remove('copy-success');
            }}, 2000);
        }};
        
        window.showCopyError_{instance_id} = function() {{
            const copyButton = document.getElementById('copy-selected-btn-{instance_id}');
            const originalText = copyButton.textContent;
            
            // Change button text to show error
            copyButton.textContent = '✗ Copy Failed';
            copyButton.classList.add('copy-error');
            
            // Reset after delay
            setTimeout(() => {{
                copyButton.textContent = originalText;
                copyButton.classList.remove('copy-error');
            }}, 2000);
        }};
        
        window.searchFiles_{instance_id} = function() {{
            const input = document.getElementById('file-search-input-{instance_id}');
            const filter = input.value.toLowerCase();
            const fileExplorer = document.getElementById('file-explorer-{instance_id}');
            
            if (!filter) {{
                // If search is cleared, reset display properties and collapse folders
                // that should be collapsed by default
                const allFolders = fileExplorer.querySelectorAll('.folder');
                allFolders.forEach(folder => {{
                    // Check if this folder is in the common prefixes (should be open)
                    if (folder.classList.contains('auto-expand')) {{
                        folder.classList.add('open');
                        folder.querySelector('.folder-toggle').textContent = '▼';
                        folder.querySelector('.folder-icon').textContent = '📂';
                    }} else {{
                        folder.classList.remove('open');
                        folder.querySelector('.folder-toggle').textContent = '▶';
                        folder.querySelector('.folder-icon').textContent = '📁';
                    }}
                }});
                
                // Show all file items
                const allFiles = fileExplorer.querySelectorAll('.file-item');
                allFiles.forEach(file => {{
                    file.style.display = '';
                }});
                
                return;
            }}
            
            // Track folders that contain matches
            const foldersWithMatches = new Set();
            
            // Search through all file items - now including permissions search
            const fileItems = fileExplorer.querySelectorAll('.file-item');
            fileItems.forEach(file => {{
                const filePath = file.getAttribute('data-path');
                const permissions = file.getAttribute('data-permissions') || '';
                
                // Check if the filter matches either the file path OR permissions
                const foundInPath = filePath.toLowerCase().includes(filter);
                const foundInPermissions = permissions.toLowerCase().includes(filter);
                const found = foundInPath || foundInPermissions;
                
                // If found, highlight the matching permission if it's a permission match
                if (foundInPermissions && !foundInPath) {{
                    // Add highlight class to matching permission pills
                    const userPills = file.querySelectorAll('.user-pill');
                    userPills.forEach(pill => {{
                        if (pill.textContent.toLowerCase().includes(filter)) {{
                            pill.classList.add('highlight-match');
                        }} else {{
                            pill.classList.remove('highlight-match');
                        }}
                    }});
                }} else {{
                    // Remove any existing highlights
                    const userPills = file.querySelectorAll('.user-pill');
                    userPills.forEach(pill => pill.classList.remove('highlight-match'));
                }}
                
                file.style.display = found ? '' : 'none';
                
                if (found) {{
                    // Add all parent folders to the set of folders with matches
                    let parent = file.closest('.folder');
                    while (parent) {{
                        foldersWithMatches.add(parent);
                        parent = parent.parentElement.closest('.folder');
                    }}
                }}
            }});
            
            // Open all folders with matches, close others - unchanged
            const allFolders = fileExplorer.querySelectorAll('.folder');
            allFolders.forEach(folder => {{
                if (foldersWithMatches.has(folder)) {{
                    folder.classList.add('open');
                    folder.querySelector('.folder-toggle').textContent = '▼';
                    folder.querySelector('.folder-icon').textContent = '📂';
                }} else {{
                    // Check if this folder contains any other folders with matches
                    const hasMatchingChild = Array.from(folder.querySelectorAll('.folder'))
                        .some(child => foldersWithMatches.has(child));
                        
                    if (hasMatchingChild) {{
                        folder.classList.add('open');
                        folder.querySelector('.folder-toggle').textContent = '▼';
                        folder.querySelector('.folder-icon').textContent = '📂';
                    }} else {{
                        folder.classList.remove('open');
                        folder.querySelector('.folder-toggle').textContent = '▶';
                        folder.querySelector('.folder-icon').textContent = '📁';
                    }}
                }}
            }});
        }};
        
        // Initialize on DOMContentLoaded
        document.addEventListener('DOMContentLoaded', function() {{
            // Apply initial state to auto-expanded folders
            const fileExplorer = document.getElementById('file-explorer-{instance_id}');
            if (fileExplorer) {{
                const autoExpandFolders = fileExplorer.querySelectorAll('.folder.open');
                autoExpandFolders.forEach(folder => {{
                    folder.classList.add('auto-expand');
                }});
            }}
        }});
    }})();
    </script>
    """
    
    # Also add the highlight style to the CSS
    css_code = css_code.replace("</style>", """
    /* Search highlight style */
    .highlight-match {
        background-color: #ffffc0 !important;
        border-color: #e6e600 !important;
        color: #333 !important;
        font-weight: bold;
    }
    </style>
    """)
    
    # Full HTML with unique IDs for this instance
    html = f"""
    {css_code}
    <div id="file-container-{instance_id}">
        <div id="file-header-{instance_id}">
            <h3>Files in {datasite_email} ({len(files)} files)</h3>
        </div>
        <div id="file-search-controls-{instance_id}">
            <div id="file-search-{instance_id}">
                <input type="text" id="file-search-input-{instance_id}" placeholder="Search files..." onkeyup="searchFiles_{instance_id}()">
            </div>
            <div id="copy-controls-{instance_id}">
                <span id="selected-info-{instance_id}"><span id="selected-count-{instance_id}">0</span> selected</span>
                <button id="copy-selected-btn-{instance_id}" onclick="copySelectedToPythonList_{instance_id}()" disabled>
                    Copy as Python List
                </button>
            </div>
        </div>
        <div id="file-explorer-{instance_id}" class="file-explorer">
            {file_tree_html}
        </div>
    </div>
    {js_code}
    """
    
    return html 