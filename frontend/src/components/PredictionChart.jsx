import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/Card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/Tabs";

const barData = (prediction) => [
  { name: prediction.teamA, probability: prediction.winA, fill: "#1E90FF" },
  // { name: "score", probability: prediction.score, fill: "#4A4A4A" },
  { name: prediction.teamB, probability: prediction.winB, fill: "#32CD32" },
];

const pieData = (prediction) => [
  { name: prediction.teamA, value: prediction.winA, fill: "#1E90FF" },
  // { name: "score", value: prediction.score, fill: "#4A4A4A" },
  { name: prediction.teamB, value: prediction.winB, fill: "#32CD32" },
];

export default function PredictionChart({ prediction }) {
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card border rounded-lg p-3 shadow-lg">
          <p className="font-medium">{label}</p>
          <p className="text-primary">
            Probability: {payload[0].value}%
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Prediction Results</CardTitle>
        <div className="flex items-center space-x-2">
          <span className="text-muted-foreground">Confidence Score:</span>
          <span className="font-semibold text-secondary">{prediction.confidence}%</span>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="pie" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="pie">Pie Chart</TabsTrigger>
            <TabsTrigger value="bar">Bar Chart</TabsTrigger>
          </TabsList>
          
          <TabsContent value="pie" className="space-y-4">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData(prediction)}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieData(prediction).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>

          <TabsContent value="bar" className="space-y-4">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barData(prediction)} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis 
                    dataKey="name" 
                    tick={{ fontSize: 12 }}
                    interval={0}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis 
                    tick={{ fontSize: 12 }}
                    label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="probability" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
          
         
        </Tabs>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="text-center p-4 bg-blue-50  rounded-lg">
            <div className="text-2xl font-bold text-primary">{prediction.winA}%</div>
            <div className="text-sm text-muted-foreground">{prediction.teamA} Win</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-800">{prediction.score}</div>
            <div className="text-sm text-muted-foreground">Predicted Score</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-secondary">{prediction.winB}%</div>
            <div className="text-sm text-muted-foreground">{prediction.teamB} Win</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}