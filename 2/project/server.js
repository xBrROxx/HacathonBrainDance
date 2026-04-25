// server.js
const express = require('express');
const fs = require('fs').promises;
const path = require('path');

const app = express();
const PORT = 3000; // You can change this to any port, like 3000

// This serves your static files (HTML, JS, CSS, and music)
// Serve static files relative to THIS file's location, not the CWD
app.use(express.static(__dirname));

// This is the 'bridge' your front-end will call to get a song list
app.get('/api/songs/:emotion', async (req, res) => {
    const emotionFolder = path.join(__dirname, 'music', req.params.emotion);
    try {
        const files = await fs.readdir(emotionFolder);
        // Filter for .mp3 files only
        const mp3Files = files.filter(file => file.toLowerCase().endsWith('.mp3'));
        res.json(mp3Files);
    } catch (error) {
        console.error(`Error reading folder ${emotionFolder}:`, error);
        // If the folder doesn't exist or is empty, send an empty list
        res.json([]);
    }
});

app.listen(PORT, () => {
    console.log(`NeuroBeats server is running`);
    console.log(`👉 Open your browser to: http://localhost:${PORT}`);
});