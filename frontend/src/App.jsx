import { useState, useEffect } from "react";
import  Header  from "./components/Header";
import  Footer  from "./components/Footer";
import  TeamSelector  from "./components/TeamSelector";
import  PredictionChart  from "./components/PredictionChart";
import  HistorySection  from "./components/HistorySection";
import { Toaster } from "./components/ui/Sonner";
import { toast } from "sonner";
import axios from 'axios';

const baseURL = 'http://127.0.0.1:8000'

export default function App() {
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([
    {
      id: "1",
      date: "2025-01-15",
      teamA: "Manchester United",
      teamB: "Liverpool",
      predictedResult: "Liverpool Win (45%)",
      actualResult: "Liverpool 2-1",
      confidence: 78,
      status: "correct"
    },
    {
      id: "2", 
      date: "2025-01-10",
      teamA: "Arsenal",
      teamB: "Chelsea",
      predictedResult: "Arsenal Win (52%)",
      actualResult: "Draw 1-1",
      confidence: 65,
      status: "incorrect"
    },
    {
      id: "3",
      date: "2025-01-05",
      teamA: "Manchester City",
      teamB: "Tottenham",
      predictedResult: "Manchester City Win (68%)",
      actualResult: "Manchester City 3-0",
      confidence: 85,
      status: "correct"
    }
  ]);

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  const generatePrediction = (teamA, teamB) => {
    // Mock ML prediction algorithm
    const teamStrengths = {
      "Manchester City": 0.9,
      "Arsenal": 0.85,
      "Liverpool": 0.83,
      "Manchester United": 0.75,
      "Chelsea": 0.73,
      "Newcastle": 0.70,
      "Tottenham": 0.68,
      "Brighton": 0.65,
      "Aston Villa": 0.63,
      "West Ham": 0.60,
      "Leicester City": 0.55,
      "Everton": 0.52,
      "Crystal Palace": 0.50,
      "Brentford": 0.48,
      "Fulham": 0.46,
      "Wolves": 0.44,
      "Southampton": 0.40,
      "Nottingham Forest": 0.38,
      "Bournemouth": 0.36,
      "Sheffield United": 0.30
    };

    const strengthA = teamStrengths[teamA] || 0.5;
    const strengthB = teamStrengths[teamB] || 0.5;
    
    // Calculate base probabilities with some randomness
    const randomFactor = 0.1;
    const strengthDiff = strengthA - strengthB;
    
    let winA = 50 + (strengthDiff * 30) + (Math.random() - 0.5) * randomFactor * 100;
    let winB = 50 - (strengthDiff * 30) + (Math.random() - 0.5) * randomFactor * 100;
    let draw = 25 + (Math.random() - 0.5) * 10;
    
    // Normalize to 100%
    const total = winA + winB + draw;
    winA = Math.round((winA / total) * 100);
    winB = Math.round((winB / total) * 100);
    draw = 100 - winA - winB;
    
    // Ensure all values are positive
    winA = Math.max(5, winA);
    winB = Math.max(5, winB);
    draw = Math.max(5, draw);
    
    // Calculate confidence based on strength difference
    const confidence = Math.round(70 + Math.abs(strengthDiff) * 30 + Math.random() * 10);

    return {
      teamA,
      teamB,
      winA,
      winB,
      draw,
      confidence: Math.min(confidence, 95)
    };
  };

  const handlePredict = async (teamA, teamB) => {
    setIsLoading(true);

    try {
      const response = await axios.post(baseURL + '/predict/', {
        "home_team": teamA,
        "away_team": teamB
      })

      const data = response.data;
  
      // Mapping backend → format PredictionChart
      const totalGoals = data.home_pred + data.away_pred || 1;
      const mappedPrediction = {
        teamA: data.home_team,
        teamB: data.away_team,
        winA: Math.round((data.home_pred / totalGoals) * 100),
        winB: Math.round((data.away_pred / totalGoals) * 100),
        score: `${data.home_pred} - ${data.away_pred}`,
        confidence: 90 // bisa diganti perhitungan asli nanti
      };
      
      setPrediction(mappedPrediction);

      setHistory(prev => [
        {
          id: Date.now().toString(),
          date: new Date().toISOString().slice(0, 10),
          teamA: mappedPrediction.teamA,
          teamB: mappedPrediction.teamB,
          predictedResult: `${mappedPrediction.teamA} Win (${mappedPrediction.winA}%)`,
          actualResult: null,
          confidence: mappedPrediction.confidence,
          status: "pending"
        },
        ...prev // supaya urutan terbaru di atas
      ]);

      setIsLoading(false)
    } catch (error) {
      console.log(error);
    }


    
    // // Simulate ML processing time
    // await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 1000));
    
    // const newPrediction = generatePrediction(teamA, teamB);
    // setPrediction(newPrediction);
    
    // // Add to history
    // const mostLikelyResult = newPrediction.winA > newPrediction.winB 
    //   ? newPrediction.winA > newPrediction.draw 
    //     ? `${teamA} Win (${newPrediction.winA}%)`
    //     : `Draw (${newPrediction.draw}%)`
    //   : newPrediction.winB > newPrediction.draw
    //     ? `${teamB} Win (${newPrediction.winB}%)`
    //     : `Draw (${newPrediction.draw}%)`;
    
    // const newHistoryItem = {
    //   id: Date.now().toString(),
    //   date: new Date().toISOString().split('T')[0],
    //   teamA,
    //   teamB,
    //   predictedResult: mostLikelyResult,
    //   confidence: newPrediction.confidence,
    //   status: "pending"
    // };
    
    // setHistory(prev => [newHistoryItem, ...prev]);
    // setIsLoading(false);
    
    toast.success("Prediction generated successfully!");
  };

  const handleClear = () => {
    setPrediction(null);
    toast.info("Inputs cleared");
  };

  const toggleDarkMode = () => {
    // setIsDarkMode(!isDarkMode);
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header isDarkMode={isDarkMode} toggleDarkMode={toggleDarkMode} />
      
      <main className="flex-1 container mx-auto px-4 py-8 space-y-8">
        <div className="text-center space-y-2">
          <h2 className="text-muted-foreground">
            Predict football match outcomes using advanced machine learning
          </h2>
        </div>
        
        <TeamSelector 
          onPredict={handlePredict}
          onClear={handleClear}
          isLoading={isLoading}
        />
        
        {prediction && (
          <PredictionChart prediction={prediction} />
        )}
        
        <HistorySection history={history} />
      </main>
      
      <Footer />
      <Toaster />
    </div>
  );
}