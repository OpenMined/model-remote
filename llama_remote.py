from __future__ import annotations

import httpx
import json
import argparse
import sys
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import os
import glob
from pathlib import Path
import time
import pandas as pd

from loguru import logger
from pydantic import BaseModel, Field, validator, field_validator
from syft_event.types import Request
from syft_rpc import rpc

from syft_rpc_client import SyftRPCClient
from file_display import FileList, generate_files_html  # Import our new module


# ----------------- Request/Response Models -----------------

class OllamaRequest(BaseModel):
    """Request to send to a remote Ollama instance."""
    model: str = Field(description="Name of the Ollama model to use")
    prompt: str = Field(description="The prompt text to send to the model")
    system: Optional[str] = Field(default=None, description="Optional system prompt")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), 
                         description="Timestamp of the request")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Additional Ollama options")
    file_paths: Optional[List[str]] = Field(default=None, description="Paths to files to include in context")


class OllamaResponse(BaseModel):
    """Response from a remote Ollama instance."""
    model: str = Field(description="Model that generated the response")
    response: str = Field(description="Generated text response")
    error: Optional[str] = Field(default=None, description="Error message, if any")
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), 
                         description="Timestamp of the response")
    total_duration_ms: Optional[int] = Field(default=None, description="Processing time in milliseconds")
    
    @field_validator('error')
    def check_error(cls, v, info):
        if v and not info.data.get('response'):
            info.data['response'] = f"Error: {v}"
        return v


# Define a specific request class that extends OllamaRequest to trigger file listing
class FileListRequest(OllamaRequest):
    """Special request to list accessible files instead of generating text."""
    model: str = Field(default="file_list", description="Fixed value to identify file list request")
    prompt: str = Field(default="list_files", description="Fixed value to identify file list request")
    list_for_user: str = Field(description="Email of the user to list files for")


# Missing model definition - add this with the other models
class FilePermissionRequest(BaseModel):
    """Request to list files accessible to a user."""
    user_email: str = Field(description="Email of the user to check permissions for")
    file_paths: List[str] = Field(default_factory=list, description="List of file paths (empty for listing all)")
    operation: str = Field(default="list", description="Operation: 'list' to get all permitted files")
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc),
                         description="Timestamp of the request")


class FilePermissionResponse(BaseModel):
    """Response containing files a user has permission to access."""
    user_email: str = Field(description="Email of the user permissions were checked for")
    allowed_files: List[str] = Field(description="List of files the user can access")
    success: bool = Field(description="Whether the permission check was successful")
    error: Optional[str] = Field(default=None, description="Error message, if any")
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc),
                         description="Timestamp of the response")


# ----------------- Ollama Client Implementation -----------------

class OllamaClient(SyftRPCClient):
    """Client for sending prompts to remote Ollama instances."""
    
    def __init__(self, 
                 config_path: Optional[str] = None, 
                 ollama_url: str = "http://localhost:11434",
                 start_server: bool = False):
        """Initialize the Ollama client.
        
        Args:
            config_path: Optional path to a custom config.json file
            ollama_url: URL of the local Ollama instance, if applicable
            start_server: Whether to start the server in a background thread
        """
        super().__init__(
            config_path=config_path,
            app_name="ollama_remote",
            endpoint="/generate",
            request_model=OllamaRequest,
            response_model=OllamaResponse,
            start_server=start_server
        )
        self.ollama_url = ollama_url
        self._files_cache = None
        self._files_cache_time = 0
        
    def _create_server(self):
        """Create and configure the server with all endpoints."""
        box = super()._create_server()
        self.box = box
        
        # Register separate endpoint for file permissions
        @box.on_request("/list_permissions")
        def permissions_handler(request_data: dict, ctx: Request) -> dict:
            try:
                # Convert the data to a FilePermissionRequest
                if isinstance(request_data, dict):
                    request = FilePermissionRequest(**request_data)
                else:
                    request = FilePermissionRequest.model_validate(request_data)
                    
                # Get the user's allowed files
                user_email = request.user_email
                allowed_files = self._get_user_allowed_files(user_email)
                
                # Create and return the response
                response = FilePermissionResponse(
                    user_email=user_email,
                    allowed_files=allowed_files,
                    success=True,
                    ts=datetime.now(timezone.utc)
                )
                
                # Convert to dict for serialization
                if hasattr(response, "model_dump"):
                    return response.model_dump(exclude_none=True, mode='json')
                elif hasattr(response, "dict"):
                    return response.dict(exclude_none=True)
                else:
                    return response
                
            except Exception as e:
                logger.error(f"Error handling file permission request: {e}")
                response = FilePermissionResponse(
                    user_email=request_data.get("user_email", "unknown"),
                    allowed_files=[],
                    success=False,
                    error=str(e),
                    ts=datetime.now(timezone.utc)
                )
                
                if hasattr(response, "model_dump"):
                    return response.model_dump(exclude_none=True, mode='json')
                elif hasattr(response, "dict"):
                    return response.dict(exclude_none=True)
                else:
                    return response
                
        return box
        
    def _check_file_permission(self, user_email: str, file_path: str) -> bool:
        """Check if a user has permission to access a file using both .syftperm_exe files
        and standard Syft permissions."""
        try:
            # First, check for .syftperm_exe specific permissions
            path = Path(file_path)
            perm_file = path.parent / f"{path.name}.syftperm_exe"
            
            logger.debug(f"Checking permissions for user {user_email} on file {file_path}")
            
            # If .syftperm_exe file exists, check its permissions
            if perm_file.exists():
                logger.debug(f"Found permission file: {perm_file}")
                try:
                    with open(perm_file, 'r') as f:
                        permissions = json.load(f)
                        
                    # Check if user is in allowed_users or if "*" is in allowed_users (wildcard for any user)
                    allowed_users = permissions.get("allowed_users", [])
                    logger.debug(f"Users with explicit permission: {allowed_users}")
                    if user_email in allowed_users or "*" in allowed_users:
                        logger.debug(f"User {user_email} has explicit permission")
                        return True
                except Exception as e:
                    logger.error(f"Error reading .syftperm_exe file {perm_file}: {e}")
            
            # Fall back to checking standard Syft permissions
            logger.debug(f"No explicit permission found, checking standard Syft permissions")
            
            # Check if we can determine the datasite path
            if hasattr(self, 'client') and hasattr(self.client, 'datasite_path'):
                try:
                    # Get the datasite path
                    datasite_path = Path(self.client.datasite_path)
                    logger.debug(f"Datasite path: {datasite_path}")
                    
                    # Get the relative path - manually since relative_to might fail if not a subdirectory
                    file_path_str = str(path)
                    datasite_path_str = str(datasite_path)
                    
                    if file_path_str.startswith(datasite_path_str):
                        relative_path = file_path_str[len(datasite_path_str):].lstrip('/')
                        logger.debug(f"Relative path: {relative_path}")
                        
                       # Check if the client has the has_permission method
                        if hasattr(self.box.client, 'has_permission'):
                            try:
                                has_access = self.box.client.has_permission(
                                    user=user_email, 
                                    path=relative_path, 
                                    permission="read"
                                )
                                logger.debug(f"Standard permission check result: {has_access}")
                                return has_access
                            except Exception as e:
                                logger.warning(f"Error calling has_permission: {e}")
                        else:
                            print(type(self.box.client))
                            print(self.box.client)
                            print(dir(self.box.client))

                            logger.warning("Client does not have has_permission method")
                    else:
                        logger.warning(f"File {file_path} is not within datasite path {datasite_path}")
                except Exception as e:
                    logger.warning(f"Error checking standard permissions: {e}")
            else:
                logger.warning("Client or datasite_path not available for standard permission check")
                
            # Check if user is the datasite owner
            try:
                datasite_path = self.client.datasite_path
                path_str = str(datasite_path)
                parts = path_str.split('/')
                
                # Try to extract datasite owner from path
                datasite_owner = None
                for part in parts:
                    if '@' in part:
                        datasite_owner = part
                        break
                
                if datasite_owner and user_email == datasite_owner:
                    logger.debug(f"User {user_email} is the datasite owner, granting access")
                    return True
            except Exception as e:
                logger.warning(f"Error checking if user is datasite owner: {e}")
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking file permission: {e}")
            return False
    
    def _get_user_allowed_files(self, user_email: str) -> List[str]:
        """Get list of files a user has permission to access."""
        try:
            # Get the datasite path
            if not hasattr(self, 'client') or not hasattr(self.client, 'datasite_path'):
                logger.warning("Could not determine datasite path for permission check")
                return []
            
            datasite_path = self.client.datasite_path
            allowed_files = []
            
            # Walk the datasite path and check each file for permissions
            for root, _, files in os.walk(datasite_path):
                for file in files:
                    # Skip certain file types
                    if file.endswith('.syftperm_exe') or file.endswith('.request') or \
                       file.endswith('.response') or file == "rpc.schema.json":
                        continue
                    
                    file_path = os.path.join(root, file)
                    if self._check_file_permission(user_email, file_path):
                        # Store normalized absolute path
                        norm_path = os.path.normpath(os.path.abspath(file_path))
                        allowed_files.append(norm_path)
            
            return allowed_files
            
        except Exception as e:
            logger.error(f"Error getting allowed files: {e}")
            return []
    
    def _handle_request(self, request, ctx: Request, box) -> OllamaResponse:
        """Process an incoming request."""
        
        # Check if this is a file listing request
        if hasattr(request, 'model') and request.model == "__file_listing__" and \
           hasattr(request, 'prompt') and request.prompt.startswith("list_accessible_files_for:"):
            
            # Extract the user email
            user_email = request.prompt.replace("list_accessible_files_for:", "").strip()
            logger.info(f"🔖 RECEIVED: File listing request for user '{user_email}'")
            
            # Get list of allowed files for the user
            try:
                allowed_files = self._get_user_allowed_files(user_email)
                file_list_text = "\n".join(allowed_files) if allowed_files else "No accessible files found"
                
                return OllamaResponse(
                    model="__file_listing__",
                    response=file_list_text,
                    ts=datetime.now(timezone.utc)
                )
            except Exception as e:
                logger.error(f"Error processing file listing request: {e}")
                return OllamaResponse(
                    model="__file_listing__",
                    response=f"Error listing files: {str(e)}",
                    error=str(e),
                    ts=datetime.now(timezone.utc)
                )
        
        # Check if this is a model listing request
        if hasattr(request, 'model') and request.model == "__list_models__" and \
           hasattr(request, 'prompt') and request.prompt == "list_available_models":
            
            logger.info(f"🔔 RECEIVED: Model listing request")
            
            # List available models
            try:
                models_list = self.list_available_models()
                # Extract just the model names
                model_names = [model.get("name") for model in models_list if model.get("name")]
                
                # Return as a JSON string
                return OllamaResponse(
                    model="__list_models__",
                    response=json.dumps(model_names),
                    ts=datetime.now(timezone.utc)
                )
            except Exception as e:
                logger.error(f"Error processing model listing request: {e}")
                return OllamaResponse(
                    model="__list_models__",
                    response="[]",  # Empty JSON array
                    error=str(e),
                    ts=datetime.now(timezone.utc)
                )
        
        # Normal Ollama request handling
        logger.info(f"🔔 RECEIVED: Ollama request for model '{request.model}'")
        
        try:
            # Process file paths if provided and add to prompt
            modified_prompt = request.prompt
            
            if request.file_paths and len(request.file_paths) > 0:
                logger.info(f"Processing {len(request.file_paths)} files for context")
                file_contents = []
                
                # Extract user email from context (if available)
                user_email = ctx.email if hasattr(ctx, 'email') else None
                if not user_email and hasattr(ctx, 'from_email'):
                    user_email = ctx.from_email
                if not user_email:
                    user_email = self.client.email  # Fall back to client's email
                
                logger.info(f"Request from user: {user_email}")
                
                # Get all files the user has permission to access first
                allowed_files = set(self._get_user_allowed_files(user_email))
                logger.info(f"User has access to {len(allowed_files)} files")
                
                # Process each requested file
                for file_path in request.file_paths:
                    try:
                        # Check if file exists
                        if not os.path.exists(file_path):
                            logger.warning(f"File not found: {file_path}")
                            file_contents.append(f"ERROR: File not found - {file_path}")
                            continue
                        
                        # Check if the file is in the allowed files list
                        # Normalize to absolute path with consistent format
                        norm_path = os.path.normpath(os.path.abspath(file_path))
                        
                        # Debug logging to diagnose permission issues
                        logger.debug(f"Checking if {norm_path} is in allowed files set")
                        if norm_path not in allowed_files:
                            # Log a sample of allowed files for debugging
                            sample_allowed = list(allowed_files)[:5] if len(allowed_files) > 5 else list(allowed_files)
                            logger.warning(f"User {user_email} does not have permission to access {file_path}")
                            logger.debug(f"Sample of allowed files: {sample_allowed}")
                            file_contents.append(f"ERROR: Permission denied - {file_path}")
                            continue
                        else:
                            logger.debug(f"Permission granted for {norm_path}")
                        
                        # Read the file content
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Get the file extension for syntax highlighting
                        _, ext = os.path.splitext(file_path)
                        lang = ext[1:] if ext else ""  # Remove the dot
                        
                        # Format the file content with the file path and language
                        formatted_content = f"File: {file_path}\n```{lang}\n{content}\n```\n\n"
                        file_contents.append(formatted_content)
                        
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {e}")
                        file_contents.append(f"ERROR: Could not read file {file_path} - {str(e)}")
                
                # Add file contents before the prompt
                if file_contents:
                    files_text = "# Included Files:\n\n" + "\n".join(file_contents)
                    modified_prompt = f"{files_text}\n\n# User Query:\n{request.prompt}"
            
            # Prepare the request payload for Ollama
            payload = {
                "model": request.model,
                "prompt": modified_prompt,
                "stream": False,  # Ensure we're not getting a streaming response
            }
            
            # Add optional parameters
            if request.system:
                payload["system"] = request.system
            if request.temperature is not None:
                payload["temperature"] = request.temperature
            if request.max_tokens is not None:
                payload["max_tokens"] = request.max_tokens
            if request.options:
                payload.update(request.options)
                
            # Send request to the local Ollama instance
            response = httpx.post(
                f"{self.ollama_url}/api/generate", 
                json=payload,
                timeout=120.0  # Longer timeout for LLM generation
            )
            
            if response.status_code == 200:
                # Improved JSON parsing to handle different response formats
                try:
                    # Try to parse as normal JSON first
                    data = response.json()
                except json.JSONDecodeError as e:
                    # If that fails, try to extract the first valid JSON object
                    try:
                        text = response.text
                        # Find the first complete JSON object
                        json_start = text.find('{')
                        json_end = text.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            data = json.loads(text[json_start:json_end])
                        else:
                            raise ValueError(f"Could not find valid JSON in response: {text[:100]}...")
                    except Exception as nested_e:
                        return OllamaResponse(
                            model=request.model,
                            response="",
                            error=f"JSON parsing error: {str(e)}. Nested error: {str(nested_e)}",
                            ts=datetime.now(timezone.utc)
                        )
                
                # Extract and return the response
                return OllamaResponse(
                    model=request.model,
                    response=data.get("response", ""),
                    total_duration_ms=data.get("total_duration", 0),
                    ts=datetime.now(timezone.utc)
                )
            else:
                return OllamaResponse(
                    model=request.model,
                    response="",
                    error=f"HTTP Error {response.status_code}: {response.text}",
                    ts=datetime.now(timezone.utc)
                )
        except Exception as e:
            logger.error(f"Error processing Ollama request: {e}")
            return OllamaResponse(
                model=request.model,
                response="",
                error=str(e),
                ts=datetime.now(timezone.utc)
            )
    
    def generate(self, 
                 to_email: str, 
                 model: str, 
                 prompt: str, 
                 system: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 file_paths: Optional[List[str]] = None) -> Optional[OllamaResponse]:
        """Send a generation request to a remote Ollama instance.
        
        Args:
            to_email: Email of the datasite hosting the Ollama instance
            model: Name of the LLM model to use
            prompt: The prompt text to send
            system: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            file_paths: Optional list of file paths to include in context
            
        Returns:
            OllamaResponse with the generated text if successful, None otherwise
        """
        request = OllamaRequest(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            file_paths=file_paths,
            ts=datetime.now(timezone.utc)
        )
        
        return self.send_request(to_email, request)
    
    def find_files(self, pattern: str) -> List[str]:
        """Find files matching a pattern.
        
        Args:
            pattern: A glob pattern like "*.py" or a specific file path
            
        Returns:
            List of file paths matching the pattern
        """
        return glob.glob(pattern)
    
    def list_accessible_files(self, datasite_email: str) -> List[str]:
        """List files on the remote datasite that the current user has permission to access."""
        if not self._valid_datasite(datasite_email):
            logger.error(f"Invalid datasite: {datasite_email}")
            return []
        
        logger.info(f"Requesting list of accessible files from {datasite_email}")
        try:
            # Create a special Ollama request that signals file listing
            request = OllamaRequest(
                model="__file_listing__",  # Special marker model name
                prompt=f"list_accessible_files_for:{self.client.email}",
                ts=datetime.now(timezone.utc)
            )
            
            # Send as a normal Ollama request
            response = self.send_request(to_email=datasite_email, request_data=request)
            
            if not response or not hasattr(response, 'response'):
                logger.warning("Failed to get list of accessible files: Empty response")
                return []
            
            # Response contains newline-separated list of files
            file_list = response.response.strip().split('\n')
            
            # Filter out empty lines and "No accessible files found" message
            if len(file_list) == 1 and (
                file_list[0] == "No accessible files found" or
                not file_list[0].strip()
            ):
                return []
            
            return file_list
            
        except Exception as e:
            logger.error(f"Error getting accessible files: {e}")
            return []
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all models available on the local Ollama instance.
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = httpx.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                return response.json().get("models", [])
            else:
                logger.error(f"Failed to get models: HTTP {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    @property
    def files(self) -> List[str]:
        """Get list of files accessible in the client's own datasite.
        
        Returns:
            List of file paths the user has permission to access
        """
        # Cache results for 60 seconds to avoid excessive requests
        current_time = time.time()
        if self._files_cache is None or (current_time - self._files_cache_time) > 60:
            # Get all files in the client's own datasite
            try:
                datasite_path = self.client.datasite_path
                all_files = []
                
                for root, _, files in os.walk(datasite_path):
                    for file in files:
                        # Skip certain file types
                        if file.endswith('.syftperm_exe') or file.endswith('.request') or \
                           file.endswith('.response') or file == "rpc.schema.json":
                            continue
                        
                        file_path = os.path.join(root, file)
                        # Store normalized absolute path
                        norm_path = os.path.normpath(os.path.abspath(file_path))
                        all_files.append(norm_path)
                
                self._files_cache = all_files
                self._files_cache_time = current_time
            except Exception as e:
                logger.error(f"Error listing files: {e}")
                self._files_cache = []
                self._files_cache_time = current_time
        
        # Return as a FileList for nice display
        return FileList(self._files_cache, self)
    
    def _files_repr_html_(self) -> str:
        """Generate an HTML file explorer for Jupyter notebooks with permission info."""
        try:
            # Get the current list of files
            files = self._files_cache  # Use the cached files directly
            
            # Create a dictionary of file permissions
            file_permissions = {}
            for file_path in files:
                file_permissions[file_path] = self._get_file_permissions(file_path)
            
            # Use the external module to generate the HTML, passing permissions data
            return generate_files_html(files, self.client.email, file_permissions=file_permissions)
            
        except Exception as e:
            # Fallback to a simple list if anything fails
            logger.error(f"Error creating HTML file representation: {e}")
            files = self._files_cache or []
            return f"<div><p>Files in {self.client.email} ({len(files)} files):</p><ul>" + \
                   "".join([f"<li>{file}</li>" for file in files[:20]]) + \
                   (f"<li>... and {len(files) - 20} more</li>" if len(files) > 20 else "") + \
                   "</ul></div>"

    @property
    def datasites(self) -> DatasiteCollection:
        """Get the collection of available datasite clients.
        
        Returns:
            Collection of DatasiteClient objects that can be accessed by index or email
        """
        server_emails = self.list_available_servers()
        return DatasiteCollection(self, server_emails)

    def datasite(self, email: str) -> Optional[DatasiteClient]:
        """Get a datasite client by email.
        
        Args:
            email: Email of the datasite
            
        Returns:
            DatasiteClient for the specified datasite, or None if not available
        """
        if self._valid_datasite(email):
            return DatasiteClient(self, email)
        return None

    def manage_file_permissions(self,
                               file_paths: List[str],
                               add_users: Optional[List[str]] = None,
                               remove_users: Optional[List[str]] = None,
                               set_users: Optional[List[str]] = None) -> Dict[str, Any]:
        """Manage permissions for a list of files by modifying their .syftperm_exe files.
        
        Args:
            file_paths: List of file paths to modify permissions for
            add_users: Optional list of user emails to add access for
            remove_users: Optional list of user emails to remove access from
            set_users: Optional list of user emails to set as the exclusive allowed users 
                      (overrides the current permissions completely)
        
        Returns:
            Dict with results of the operation, including successes and failures
        """
        if not file_paths:
            return {"success": False, "error": "No file paths provided", "results": {}}
        
        if not any([add_users, remove_users, set_users]):
            return {"success": False, "error": "No permission changes specified", "results": {}}
        
        # Track results for each file
        results = {}
        success_count = 0
        failure_count = 0
        
        # Process each file
        for file_path in file_paths:
            try:
                file_path = os.path.normpath(os.path.abspath(file_path))
                perm_file = Path(file_path + ".syftperm_exe")
                file_path_str = str(file_path)
                
                # Initialize or load the current permissions
                if perm_file.exists():
                    try:
                        with open(perm_file, 'r') as f:
                            permissions = json.load(f)
                            current_users = permissions.get("allowed_users", [])
                    except Exception as e:
                        logger.warning(f"Error reading {perm_file}: {e}. Creating new permissions file.")
                        permissions = {}
                        current_users = []
                else:
                    permissions = {}
                    current_users = []
                
                # If using set_users, override completely
                if set_users is not None:
                    new_users = list(set(set_users))  # Deduplicate
                    permissions["allowed_users"] = new_users
                    operation = "set"
                else:
                    # Start with current users
                    new_users = list(current_users)
                    
                    # Add users if specified
                    if add_users:
                        for user in add_users:
                            if user not in new_users:
                                new_users.append(user)
                        operation = "add"
                    
                    # Remove users if specified
                    if remove_users:
                        new_users = [u for u in new_users if u not in remove_users]
                        operation = "remove" if not add_users else "modify"
                    
                    # Update permissions
                    permissions["allowed_users"] = new_users
                
                # Write the updated permissions file
                try:
                    # Ensure parent directory exists
                    os.makedirs(os.path.dirname(perm_file), exist_ok=True)
                    
                    with open(perm_file, 'w') as f:
                        json.dump(permissions, f, indent=2)
                    
                    success_count += 1
                    results[file_path_str] = {
                        "success": True,
                        "operation": operation,
                        "allowed_users": new_users,
                        "perm_file": str(perm_file)
                    }
                    logger.info(f"Updated permissions for {file_path}")
                    
                except Exception as e:
                    failure_count += 1
                    results[file_path_str] = {
                        "success": False,
                        "error": str(e)
                    }
                    logger.error(f"Failed to update permissions for {file_path}: {e}")
                    
            except Exception as e:
                failure_count += 1
                results[file_path_str] = {
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"Error processing {file_path}: {e}")
        
        # Return the final summary
        return {
            "success": failure_count == 0,
            "total": len(file_paths),
            "successful": success_count,
            "failed": failure_count,
            "results": results
        }

    def set_permissions_for_selected(self, 
                                    file_list: List[str], 
                                    allowed_users: List[str]) -> Dict[str, Any]:
        """Set permissions for a list of selected files.
        
        This is a convenience method for use with the file picker UI.
        
        Args:
            file_list: List of selected file paths
            allowed_users: List of user emails to grant access to
            
        Returns:
            Dict with results of the operation
        """
        return self.manage_file_permissions(
            file_paths=file_list,
            set_users=allowed_users
        )

    def _get_file_permissions(self, file_path: str) -> List[str]:
        """Get the list of users with permission to access a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of user emails with access permissions
        """
        try:
            perm_file = Path(file_path + ".syftperm_exe")
            
            # Check if permission file exists
            if perm_file.exists():
                try:
                    with open(perm_file, 'r') as f:
                        permissions = json.load(f)
                        allowed_users = permissions.get("allowed_users", [])
                        return allowed_users
                except Exception as e:
                    logger.error(f"Error reading permissions file {perm_file}: {e}")
                    return []
            else:
                # If no specific permissions, default to owner only
                try:
                    # Try to extract owner from datasite path
                    datasite_path = self.client.datasite_path
                    path_str = str(datasite_path)
                    parts = path_str.split('/')
                    
                    # Extract datasite owner from path
                    datasite_owner = None
                    for part in parts:
                        if '@' in part:
                            datasite_owner = part
                            break
                    
                    return [datasite_owner] if datasite_owner else []
                except Exception:
                    return []
        except Exception as e:
            logger.error(f"Error getting permissions for {file_path}: {e}")
            return []


# ----------------- API Functions -----------------

def client(config_path: Optional[str] = None, 
           ollama_url: str = "http://localhost:11434",
           start_server: bool = False) -> OllamaClient:
    """Create and return a new Ollama client.
    
    Args:
        config_path: Optional path to a custom config.json file
        ollama_url: URL of the local Ollama instance
        start_server: Whether to start the server in a background thread
        
    Returns:
        An OllamaClient instance
    """
    return OllamaClient(config_path, ollama_url, start_server)


# Add command-line interface for standalone server mode
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a standalone Ollama Remote server")
    parser.add_argument("--config", help="Path to config.json file")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="URL of the Ollama instance")
    args = parser.parse_args()
    
    # Create a client with server only mode
    server = OllamaClient(config_path=args.config, ollama_url=args.ollama_url, start_server=False)
    server.run_standalone_server()


# Add this class before FileList class
class DatasiteClient:
    """Client for interacting with a specific datasite hosting Ollama."""
    
    def __init__(self, parent_client: OllamaClient, email: str):
        """Initialize a datasite client.
        
        Args:
            parent_client: The parent OllamaClient instance
            email: Email of the datasite
        """
        self._parent = parent_client
        self.email = email
        self._files_cache = None
        self._files_cache_time = 0
        self._models_cache = None
        self._models_cache_time = 0
    
    @property
    def files(self) -> List[str]:
        """Get the list of files accessible in this datasite.
        
        Returns:
            List of file paths the user has permission to access
        """
        # Cache results for 60 seconds to avoid excessive requests
        current_time = time.time()
        if self._files_cache is None or (current_time - self._files_cache_time) > 60:
            self._files_cache = self._parent.list_accessible_files(self.email)
            self._files_cache_time = current_time
        
        # Return as a FileList for nice display
        return FileList(self._files_cache, self)
    
    def _files_repr_html_(self) -> str:
        """Generate an HTML file explorer for Jupyter notebooks."""
        try:
            # Get the current list of files
            files = self._files_cache  # Use the cached files directly to avoid recursion
            
            # Create a dictionary of file permissions
            file_permissions = {}
            for file_path in files:
                file_permissions[file_path] = self._parent._get_file_permissions(file_path)
            
            # Use the external module to generate the HTML
            return generate_files_html(files, self.email, file_permissions=file_permissions)
            
        except Exception as e:
            # Fallback to a simple list if anything fails
            logger.error(f"Error creating HTML file representation: {e}")
            files = self._files_cache or []
            return f"<div><p>Files in {self.email} ({len(files)} files):</p><ul>" + \
                   "".join([f"<li>{file}</li>" for file in files[:20]]) + \
                   (f"<li>... and {len(files) - 20} more</li>" if len(files) > 20 else "") + \
                   "</ul></div>"
    
    @property
    def models(self) -> List[str]:
        """Get the list of available models on this datasite.
        
        Returns:
            List of model names available on the datasite
        """
        # Cache results for 60 seconds to avoid excessive requests
        current_time = time.time()
        if self._models_cache is None or (current_time - self._models_cache_time) > 60:
            # Make a special request to get available models
            try:
                request = OllamaRequest(
                    model="__list_models__",
                    prompt="list_available_models",
                    ts=datetime.now(timezone.utc)
                )
                
                response = self._parent.send_request(to_email=self.email, request_data=request)
                
                if response and hasattr(response, 'response'):
                    # Parse the response which should be a JSON string of model names
                    try:
                        models = json.loads(response.response)
                        self._models_cache = models
                    except json.JSONDecodeError:
                        # Fallback to splitting by newlines
                        self._models_cache = [x.strip() for x in response.response.split('\n') if x.strip()]
                else:
                    # Default to a safe known model if we can't get the list
                    self._models_cache = ["llama3"]
                    
                self._models_cache_time = current_time
            except Exception as e:
                logger.error(f"Error listing models from datasite {self.email}: {e}")
                # Return a safe default
                self._models_cache = ["llama3"]
                self._models_cache_time = current_time
                
        return self._models_cache
    
    def generate(self, 
                 prompt: str,
                 model: Optional[str] = None,
                 system: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 file_paths: Optional[List[str]] = None) -> Optional[OllamaResponse]:
        """Send a generation request to this datasite.
        
        Args:
            prompt: The prompt text to send
            model: Name of the LLM model to use (defaults to first available model)
            system: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            file_paths: Optional list of file paths to include in context
            
        Returns:
            OllamaResponse with the generated text if successful, None otherwise
        """
        # If no model is specified, use the first available model
        if model is None:
            available_models = self.models
            if not available_models:
                raise ValueError("No models available on this datasite")
            model = available_models[0]
            logger.info(f"Using default model: {model}")
        
        return self._parent.generate(
            to_email=self.email,
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            file_paths=file_paths
        )
    
    def __repr__(self) -> str:
        """String representation of the datasite client."""
        return f"DatasiteClient(email='{self.email}')"


# Add this class before the DatasiteClient class
class DatasiteCollection:
    """A collection of datasite clients that can be accessed by index or email."""
    
    def __init__(self, parent_client: OllamaClient, emails: List[str]):
        """Initialize the collection with datasites for the given emails.
        
        Args:
            parent_client: The parent OllamaClient instance
            emails: List of datasite emails to include
        """
        self._parent = parent_client
        self._emails = emails
        self._clients = {email: DatasiteClient(parent_client, email) for email in emails}
        
    def __getitem__(self, key):
        """Get a datasite by index or email."""
        if isinstance(key, int):
            # List-like integer indexing
            if key < 0 or key >= len(self._emails):
                raise IndexError(f"Index {key} out of range for {len(self._emails)} datasites")
            return self._clients[self._emails[key]]
        elif isinstance(key, str):
            # Dict-like string indexing
            if key not in self._clients:
                raise KeyError(f"No datasite found with email '{key}'")
            return self._clients[key]
        else:
            raise TypeError(f"Datasite index must be int or str, not {type(key).__name__}")
    
    def __iter__(self):
        """Iterate through all datasites."""
        for email in self._emails:
            yield self._clients[email]
    
    def __len__(self):
        """Get the number of datasites."""
        return len(self._emails)
    
    def __contains__(self, item):
        """Check if a datasite email is in the collection."""
        if isinstance(item, str):
            return item in self._clients
        elif isinstance(item, DatasiteClient):
            return item.email in self._clients
        return False
    
    def __repr__(self):
        """String representation of the collection."""
        return f"DatasiteCollection({len(self._emails)} datasites: {', '.join(self._emails)})"

    def _repr_html_(self):
        """Generate an HTML representation for Jupyter notebooks."""
        try:
            # Create a simple DataFrame with just datasite information
            data = []
            for email in self._emails:
                data.append({
                    "Datasite": email,
                    "Status": "Available"
                })
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Add some styling
            styled_df = df.style.set_properties(**{
                'text-align': 'left',
                'border': '1px solid #ddd',
                'padding': '8px'
            })
            
            styled_df = styled_df.set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#f2f2f2'),
                    ('color', 'black'),
                    ('font-weight', 'bold'),
                    ('text-align', 'left'),
                    ('border', '1px solid #ddd'),
                    ('padding', '8px')
                ]},
                {'selector': 'caption', 'props': [
                    ('font-size', '1.2em'),
                    ('font-weight', 'bold'),
                    ('padding', '10px'),
                    ('text-align', 'left')
                ]}
            ])
            
            # Add a caption/title
            return styled_df.set_caption(f"Available Ollama Datasites ({len(self._emails)})").to_html()
        
        except Exception as e:
            # Fallback to a simple HTML table if pandas styling fails
            html = "<table><caption>Available Datasites</caption><tr><th>Datasite</th><th>Status</th></tr>"
            for email in self._emails:
                html += f"<tr><td>{email}</td><td>Available</td></tr>"
            html += "</table>"
            return html
