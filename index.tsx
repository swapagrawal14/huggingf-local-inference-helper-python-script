/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';

const pythonCodeTemplate = (modelId, loraUrl, prompt) => `
# Python script to run a Hugging Face text-to-image model with a LoRA.
#
# INSTRUCTIONS:
# 1. Complete Steps 1-3 in the app for one-time setup.
# 2. Use Step 4 to configure your model and prompt.
# 3. If you get a 403 error, visit the model page (link in Step 4) to accept the terms.
# 4. Use Step 5 to download this script and run it.

import torch
from diffusers import FluxPipeline
from pathlib import Path
from huggingface_hub import hf_hub_download
import os

def generate_image(model_id, lora_url, prompt_text):
    """
    Generates an image using a base model and a LoRA file from a URL.
    """
    try:
        print("-> Setting up the FLUX pipeline...")
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        )

        print(f"-> Downloading & caching LoRA from: {lora_url}")
        lora_repo_id = "/".join(lora_url.split('/')[3:5])
        lora_filename = lora_url.split('/')[-1]
        
        cached_lora_path = hf_hub_download(repo_id=lora_repo_id, filename=lora_filename)

        adapter_name = Path(lora_filename).stem
        print(f"-> Applying LoRA weights with adapter name: '{adapter_name}'")
        pipe.load_lora_weights(cached_lora_path, adapter_name=adapter_name)
        pipe.fuse_lora()

        print("-> Moving model to GPU (if available)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        print(f"-> Using device: {device.upper()}")

        print("-> Generating image for prompt...")
        image = pipe(
            prompt=prompt_text,
            num_inference_steps=8,
            guidance_scale=0.0
        ).images[0]

        output_path = "generated_image.png"
        print(f"-> Saving image to {output_path}")
        image.save(output_path)

        print("\\n✨ Generation complete! Image saved as {output_path} ✨")

    except Exception as e:
        print(f"\\nAn error occurred: {e}")
        print("Please ensure you have run the setup commands and have a compatible environment (e.g., PyTorch with CUDA).")

if __name__ == "__main__":
    MODEL_ID = "${modelId}"
    LORA_URL = "${loraUrl}"
    PROMPT = """${prompt}"""

    generate_image(MODEL_ID, LORA_URL, PROMPT)
`;

function App() {
  const [model, setModel] = useState('black-forest-labs/FLUX.1-dev');
  const [loraUrl, setLoraUrl] = useState('https://huggingface.co/glif-loradex-trainer/swapagrawal14_flux_dev_swap_draws_it/resolve/main/flux_dev_swap_draws_it.safetensors');
  const [prompt, setPrompt] = useState('A robot drawing a masterpiece');
  const [copyStatus, setCopyStatus] = useState({ setup: 'Copy', auth: 'Copy', generated: 'Copy Code', prerequisites: 'Copy' });
  const [generatedCode, setGeneratedCode] = useState('');

  useEffect(() => {
    const code = pythonCodeTemplate(model, loraUrl, prompt);
    setGeneratedCode(code);
  }, [model, loraUrl, prompt]);
  
  const copyToClipboard = (text: string, key: 'setup' | 'auth' | 'generated' | 'prerequisites') => {
    navigator.clipboard.writeText(text).then(() => {
        setCopyStatus(prev => ({ ...prev, [key]: 'Copied!' }));
        setTimeout(() => {
            setCopyStatus(prev => ({ ...prev, [key]: key === 'generated' ? 'Copy Code' : 'Copy' }));
        }, 2000);
    }, () => {
        alert('Failed to copy!');
    });
  };
  
  const downloadCode = () => {
    const blob = new Blob([generatedCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'generate_image.py';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="title">Local AI Inference Helper</h1>
        <p className="subtitle">Configure your model and prompt, then get the Python code to run it on your own machine.</p>
      </header>
      
      <main className="main-content">
        <section className="prerequisites-section">
            <h2 className="section-title">Step 1: Install Prerequisites</h2>
            <p className="section-intro">
                The Hugging Face command-line tool uses Git to work correctly. If you don't have Git installed, you might see a <code>FileNotFoundError</code>. This is a one-time setup.
            </p>
            <a href="https://git-scm.com/downloads" target="_blank" rel="noopener noreferrer" className="external-link-btn">
                Download Git
            </a>
        </section>

        <section className="setup-section">
            <h2 className="section-title">Step 2: Setup Your Environment</h2>
            <p className="section-intro">Open your terminal and run this command. This only needs to be done once.</p>
            <div className="code-block-container">
                <pre className="code-block"><code>pip install torch diffusers transformers accelerate safetensors huggingface_hub</code></pre>
                <button 
                  className={`copy-btn ${copyStatus.setup === 'Copied!' ? 'copied' : ''}`} 
                  onClick={() => copyToClipboard('pip install torch diffusers transformers accelerate safetensors huggingface_hub', 'setup')}
                  disabled={copyStatus.setup === 'Copied!'}
                >
                  {copyStatus.setup}
                </button>
            </div>
        </section>

        <section className="auth-section">
            <h2 className="section-title">Step 3: Authenticate with Hugging Face (Important)</h2>
            <p className="section-intro">
                Some models, like FLUX, require you to log in. Run the command below and paste in your access token. If you need to change or update your token later, you can run this same command again.
            </p>
            <div className="code-block-container">
                <pre className="code-block"><code>hf auth login</code></pre>
                <button 
                  className={`copy-btn ${copyStatus.auth === 'Copied!' ? 'copied' : ''}`} 
                  onClick={() => copyToClipboard('hf auth login', 'auth')}
                  disabled={copyStatus.auth === 'Copied!'}
                >
                  {copyStatus.auth}
                </button>
            </div>
            <p className="prompt-note">
              After pasting your token and pressing Enter, the terminal might ask: <code>Add token as git credential? (Y/n)</code>. If you installed Git in Step 1, you can type <code>Y</code>. Otherwise, it is safe to type <code>n</code> and press Enter. Your token will still be saved correctly.
            </p>
            <details className="token-guide">
                <summary>Need help creating an access token?</summary>
                <ol>
                    <li>Go to the <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer">Hugging Face token settings page</a>.</li>
                    <li>Click the "New token" button.</li>
                    <li>Give your token a name (e.g., "AI Helper").</li>
                    <li>Choose the "read" role from the dropdown. This is important.</li>
                    <li>Click "Generate a token".</li>
                    <li>Click the copy icon next to your new token. You will paste this into the terminal after running the login command.</li>
                    <li className="important-note">When you paste your token, <strong>it will not be visible</strong> for security reasons. This is normal. Just press <strong>Enter</strong> after you paste it.</li>
                </ol>
            </details>
        </section>

        <section className="generator-section">
            <h2 className="section-title">Step 4: Configure Your Model & Prompt</h2>
            <form className="config-form">
                <div className="form-group">
                    <label htmlFor="model-input">Base Model ID (from Hugging Face)</label>
                    <input id="model-input" type="text" value={model} onChange={(e) => setModel(e.target.value)} required />
                </div>
                <div className="form-group">
                    <label htmlFor="lora-input">LoRA .safetensors URL</label>
                    <input id="lora-input" type="url" value={loraUrl} onChange={(e) => setLoraUrl(e.target.value)} required />
                </div>
                <div className="form-group">
                    <label htmlFor="prompt-input">Your Prompt</label>
                    <textarea id="prompt-input" value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3} required />
                </div>
            </form>
            <div className="access-note">
                <h4>Final Step for Gated Models</h4>
                <p>If you see a <strong>403 Forbidden</strong> error, it means the model requires you to accept its license. Click the button below, accept the terms on the model's page, then run the script again.</p>
                <a href={`https://huggingface.co/${model}`} target="_blank" rel="noopener noreferrer" className="access-note-btn">
                    Visit Model Page to Accept Terms
                </a>
            </div>
        </section>
        
        {generatedCode && (
            <section className="output-section">
                <h2 className="section-title">Step 5: Download & Run Your Script</h2>
                <p className="section-intro">
                    Your personal Python script is ready. Just download it and run it from your terminal.
                </p>
                <div className="code-block-container">
                    <pre className="code-block"><code>{generatedCode}</code></pre>
                </div>
                 <div className="action-buttons">
                    <button className="download-btn" onClick={downloadCode}>
                        Download .py
                    </button>
                    <button 
                      className={`copy-btn-secondary ${copyStatus.generated === 'Copied!' ? 'copied' : ''}`}
                      onClick={() => copyToClipboard(generatedCode, 'generated')}
                      disabled={copyStatus.generated === 'Copied!'}
                    >
                        {copyStatus.generated}
                    </button>
                </div>
                <div id="how-to-run-guide">
                    <h3>How to run it:</h3>
                    <ul>
                        <li>
                            <span>🖥️</span>
                            <div>Open your terminal or command prompt.</div>
                        </li>
                        <li>
                            <span>📂</span>
                            <div>Navigate to the folder where you downloaded the file (e.g., <code>cd Downloads</code>).</div>
                        </li>
                        <li>
                            <span>🚀</span>
                            <div>Run the script with the command: <code>python generate_image.py</code></div>
                        </li>
                    </ul>
                </div>
            </section>
        )}
      </main>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);