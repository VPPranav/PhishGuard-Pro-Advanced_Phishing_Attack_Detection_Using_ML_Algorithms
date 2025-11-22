document.getElementById("checkBtn").addEventListener("click", async () => {
    const result = document.getElementById("result");
    result.textContent = "Checking...";

    try {
        let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        let url = tab.url;

        const res = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url })
        });

        const data = await res.json();
        if (!res.ok) {
            result.textContent = `Error: ${data.detail}`;
            return;
        }

        let color =
            data.verdict === "phishing" ? "red" :
            data.verdict === "legitimate" ? "green" : "orange";

        result.innerHTML = `
            <div>Verdict: <span style="color:${color}; font-weight:bold;">${data.verdict.toUpperCase()}</span></div>
            <div style="margin-top:8px;">Confidence Score: <b>${data.confidence_score}%</b></div>
        `;

    } catch (err) {
        console.error(err);
        result.textContent = "Error: Could not reach backend";
    }
});
