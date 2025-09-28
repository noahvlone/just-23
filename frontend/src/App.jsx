import React, { useState, useEffect } from 'react';
import { 
    Container, 
    Paper, 
    TextField, 
    Button, 
    Typography, 
    Box, 
    Card, 
    CardContent,
    List,
    ListItem,
    IconButton,
    Chip,
    CircularProgress,
    ThemeProvider,
    createTheme,
    Switch,
    Grid,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Alert,
    Tabs,
    Tab
} from '@mui/material';
import { 
    Brightness4, 
    Brightness7, 
    Delete, 
    History, 
    Analytics,
    Chat as ChatIcon
} from '@mui/icons-material';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    Title,
    Tooltip,
    Legend
);

const App = () => {
    const [darkMode, setDarkMode] = useState(false);
    const [newsText, setNewsText] = useState('');
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [chatMessage, setChatMessage] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const [history, setHistory] = useState([]);
    const [activeTab, setActiveTab] = useState(0);
    const [userId] = useState('user-' + Date.now()); // Simple user ID for demo

    const theme = createTheme({
        palette: {
            mode: darkMode ? 'dark' : 'light',
            primary: {
                main: darkMode ? '#90caf9' : '#1976d2',
            },
            background: {
                default: darkMode ? '#121212' : '#f5f5f5',
                paper: darkMode ? '#1e1e1e' : '#ffffff',
            },
        },
    });

    const analyzeNews = async () => {
        if (!newsText.trim()) return;
        
        setLoading(true);
        try {
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: newsText,
                    user_id: userId
                }),
            });
            
            const data = await response.json();
            setPrediction(data);
            setChatHistory([]);
            await loadHistory();
        } catch (error) {
            console.error('Error analyzing news:', error);
        }
        setLoading(false);
    };

    const sendChatMessage = async () => {
        if (!chatMessage.trim() || !prediction) return;
        
        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prediction_id: prediction.prediction_id,
                    message: chatMessage,
                    user_id: userId
                }),
            });
            
            const data = await response.json();
            setChatHistory(prev => [...prev, {
                user: chatMessage,
                gemini: data.response,
                timestamp: new Date().toLocaleTimeString()
            }]);
            setChatMessage('');
            await loadHistory();
        } catch (error) {
            console.error('Error sending chat message:', error);
        }
    };

    const loadHistory = async () => {
        try {
            const response = await fetch(`http://localhost:8000/history/${userId}`);
            const data = await response.json();
            setHistory(data);
        } catch (error) {
            console.error('Error loading history:', error);
        }
    };

    const deleteHistoryItem = async (predictionId) => {
        try {
            await fetch(`http://localhost:8000/history/${predictionId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ user_id: userId }),
            });
            await loadHistory();
            if (prediction?.prediction_id === predictionId) {
                setPrediction(null);
                setChatHistory([]);
            }
        } catch (error) {
            console.error('Error deleting history:', error);
        }
    };

    useEffect(() => {
        loadHistory();
    }, []);

    const extractKeywords = (analysis) => {
        if (!analysis) return [];
        const words = analysis.toLowerCase().match(/\b\w+\b/g) || [];
        const keywordCount = words.reduce((acc, word) => {
            if (word.length > 3) {
                acc[word] = (acc[word] || 0) + 1;
            }
            return acc;
        }, {});
        
        return Object.entries(keywordCount)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10)
            .map(([word, count]) => ({ word, count }));
    };

    const renderAnalysisCharts = () => {
        if (!prediction) return null;

        const keywords = extractKeywords(prediction.gemini_analysis);
        
        const keywordData = {
            labels: keywords.map(k => k.word),
            datasets: [
                {
                    label: 'Keyword Frequency',
                    data: keywords.map(k => k.count),
                    backgroundColor: prediction.prediction === 'REAL' ? 
                        ['#4caf50', '#81c784', '#a5d6a7'] : 
                        ['#f44336', '#e57373', '#ef9a9a'],
                },
            ],
        };

        const confidenceData = {
            labels: ['Confidence Level'],
            datasets: [
                {
                    label: 'Prediction Confidence',
                    data: [prediction.confidence * 100],
                    backgroundColor: prediction.prediction === 'REAL' ? '#4caf50' : '#f44336',
                    borderColor: prediction.prediction === 'REAL' ? '#388e3c' : '#d32f2f',
                    borderWidth: 2,
                },
            ],
        };

        return (
            <Grid container spacing={3} sx={{ mt: 2 }}>
                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            Prediction Confidence
                        </Typography>
                        <Bar 
                            data={confidenceData}
                            options={{
                                scales: {
                                    y: {
                                        beginAtZero: true,
                                        max: 100,
                                        ticks: {
                                            callback: value => value + '%'
                                        }
                                    }
                                }
                            }}
                        />
                    </Paper>
                </Grid>
                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            Key Analysis Keywords
                        </Typography>
                        <Doughnut data={keywordData} />
                    </Paper>
                </Grid>
            </Grid>
        );
    };

    return (
        <ThemeProvider theme={theme}>
            <Container maxWidth="lg" sx={{ 
                bgcolor: 'background.default',
                minHeight: '100vh',
                py: 3
            }}>
                {/* Header */}
                <Box sx={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    mb: 4
                }}>
                    <Typography variant="h4" component="h1" color="primary">
                        🕵️ Fake News Detector
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <IconButton onClick={() => setDarkMode(!darkMode)} color="inherit">
                            {darkMode ? <Brightness7 /> : <Brightness4 />}
                        </IconButton>
                        <Typography variant="body2" sx={{ ml: 1 }}>
                            {darkMode ? 'Light Mode' : 'Dark Mode'}
                        </Typography>
                    </Box>
                </Box>

                {/* Tabs */}
                <Paper sx={{ mb: 3 }}>
                    <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)} centered>
                        <Tab icon={<Analytics />} label="Analyze News" />
                        <Tab icon={<History />} label="History" />
                    </Tabs>
                </Paper>

                {activeTab === 0 && (
                    <Grid container spacing={3}>
                        {/* Input Section */}
                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Enter News Text
                                </Typography>
                                <TextField
                                    multiline
                                    rows={10}
                                    fullWidth
                                    variant="outlined"
                                    value={newsText}
                                    onChange={(e) => setNewsText(e.target.value)}
                                    placeholder="Paste news article text here..."
                                    sx={{ mb: 2 }}
                                />
                                <Button
                                    variant="contained"
                                    onClick={analyzeNews}
                                    disabled={loading || !newsText.trim()}
                                    fullWidth
                                    size="large"
                                >
                                    {loading ? <CircularProgress size={24} /> : 'Analyze News'}
                                </Button>
                            </Paper>

                            {/* Chat Section */}
                            {prediction && (
                                <Paper sx={{ p: 3, mt: 3 }}>
                                    <Typography variant="h6" gutterBottom>
                                        <ChatIcon sx={{ mr: 1 }} />
                                        Ask Gemini for Details
                                    </Typography>
                                    <Box sx={{ mb: 2 }}>
                                        {chatHistory.map((chat, index) => (
                                            <Box key={index} sx={{ mb: 2 }}>
                                                <Chip 
                                                    label="You" 
                                                    color="primary" 
                                                    size="small" 
                                                    sx={{ mb: 1 }}
                                                />
                                                <Typography variant="body2">{chat.user}</Typography>
                                                <Chip 
                                                    label="Gemini" 
                                                    color="secondary" 
                                                    size="small" 
                                                    sx={{ mt: 1, mb: 1 }}
                                                />
                                                <Typography variant="body2">{chat.gemini}</Typography>
                                            </Box>
                                        ))}
                                    </Box>
                                    <Box sx={{ display: 'flex', gap: 1 }}>
                                        <TextField
                                            fullWidth
                                            size="small"
                                            value={chatMessage}
                                            onChange={(e) => setChatMessage(e.target.value)}
                                            placeholder="Ask about this analysis..."
                                            onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
                                        />
                                        <Button 
                                            variant="contained" 
                                            onClick={sendChatMessage}
                                            disabled={!chatMessage.trim()}
                                        >
                                            Send
                                        </Button>
                                    </Box>
                                </Paper>
                            )}
                        </Grid>

                        {/* Results Section */}
                        <Grid item xs={12} md={6}>
                            {prediction && (
                                <>
                                    <Paper sx={{ p: 3, mb: 3 }}>
                                        <Box sx={{ 
                                            display: 'flex', 
                                            justifyContent: 'space-between', 
                                            alignItems: 'center',
                                            mb: 2
                                        }}>
                                            <Typography variant="h5">
                                                Analysis Results
                                            </Typography>
                                            <Chip 
                                                label={prediction.prediction}
                                                color={prediction.prediction === 'REAL' ? 'success' : 'error'}
                                                variant="filled"
                                            />
                                        </Box>
                                        
                                        <Alert 
                                            severity={prediction.prediction === 'REAL' ? 'success' : 'error'}
                                            sx={{ mb: 2 }}
                                        >
                                            Confidence: {(prediction.confidence * 100).toFixed(2)}%
                                        </Alert>

                                        <Typography variant="h6" gutterBottom>
                                            Detailed Analysis
                                        </Typography>
                                        <Paper 
                                            variant="outlined" 
                                            sx={{ 
                                                p: 2, 
                                                bgcolor: 'background.default',
                                                maxHeight: 300,
                                                overflow: 'auto'
                                            }}
                                        >
                                            <Typography variant="body2" style={{ whiteSpace: 'pre-wrap' }}>
                                                {prediction.gemini_analysis}
                                            </Typography>
                                        </Paper>
                                    </Paper>

                                    {renderAnalysisCharts()}
                                </>
                            )}
                        </Grid>
                    </Grid>
                )}

                {activeTab === 1 && (
                    <Paper sx={{ p: 3 }}>
                        <Typography variant="h6" gutterBottom>
                            Prediction History
                        </Typography>
                        {history.length === 0 ? (
                            <Typography color="textSecondary" align="center" sx={{ py: 4 }}>
                                No prediction history yet
                            </Typography>
                        ) : (
                            <List>
                                {history.map((item) => (
                                    <ListItem key={item.id} divider>
                                        <Card sx={{ width: '100%' }}>
                                            <CardContent>
                                                <Box sx={{ 
                                                    display: 'flex', 
                                                    justifyContent: 'space-between',
                                                    alignItems: 'flex-start',
                                                    mb: 1
                                                }}>
                                                    <Box>
                                                        <Chip 
                                                            label={item.prediction_result}
                                                            color={item.prediction_result === 'REAL' ? 'success' : 'error'}
                                                            size="small"
                                                        />
                                                        <Typography variant="caption" display="block" color="textSecondary">
                                                            {new Date(item.created_at).toLocaleDateString()}
                                                        </Typography>
                                                    </Box>
                                                    <IconButton 
                                                        size="small" 
                                                        onClick={() => deleteHistoryItem(item.id)}
                                                        color="error"
                                                    >
                                                        <Delete />
                                                    </IconButton>
                                                </Box>
                                                <Typography 
                                                    variant="body2" 
                                                    sx={{ 
                                                        mb: 1,
                                                        display: '-webkit-box',
                                                        WebkitLineClamp: 2,
                                                        WebkitBoxOrient: 'vertical',
                                                        overflow: 'hidden'
                                                    }}
                                                >
                                                    {item.news_text}
                                                </Typography>
                                                <Button 
                                                    size="small" 
                                                    onClick={() => {
                                                        setPrediction({
                                                            prediction: item.prediction_result,
                                                            confidence: item.confidence,
                                                            gemini_analysis: item.gemini_analysis,
                                                            prediction_id: item.id
                                                        });
                                                        setChatHistory(item.chat_history || []);
                                                        setActiveTab(0);
                                                    }}
                                                >
                                                    View Details
                                                </Button>
                                            </CardContent>
                                        </Card>
                                    </ListItem>
                                ))}
                            </List>
                        )}
                    </Paper>
                )}
            </Container>
        </ThemeProvider>
    );
};

export default App;