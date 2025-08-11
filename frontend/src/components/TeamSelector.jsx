import { useState } from "react";
import { Button } from "./ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/Card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/SelectTeam";
import { Label } from "./ui/Label";
import { ReloadIcon } from "@radix-ui/react-icons";

const teams = [
  "Manchester United", "Manchester City", "Liverpool", "Chelsea", "Arsenal",
  "Tottenham", "Newcastle", "Brighton", "Aston Villa", "West Ham",
  "Leicester City", "Everton", "Crystal Palace", "Brentford", "Fulham",
  "Wolves", "Southampton", "Nottingham Forest", "Bournemouth", "Sheffield United"
];

function TeamSelector({ onPredict, onClear, isLoading }) {
  const [teamA, setTeamA] = useState("");
  const [teamB, setTeamB] = useState("");

  const handlePredict = () => {
    if (teamA && teamB && teamA !== teamB) {
      onPredict(teamA, teamB);
    }
  };

  const handleClear = () => {
    setTeamA("");
    setTeamB("");
    onClear();
  };

  const canPredict = teamA && teamB && teamA !== teamB && !isLoading;

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <span>⚽</span>
          <span>Select Teams</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <Label htmlFor="team-a">Team A</Label>
            <Select value={teamA} onValueChange={setTeamA}>
              <SelectTrigger id="team-a">
                <SelectValue placeholder="Select Team A" />
              </SelectTrigger>
              <SelectContent>
                {teams.map((team) => (
                  <SelectItem 
                    key={team} 
                    value={team}
                    disabled={team === teamB}
                  >
                    {team}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="team-b">Team B</Label>
            <Select value={teamB} onValueChange={setTeamB}>
              <SelectTrigger id="team-b">
                <SelectValue placeholder="Select Team B" />
              </SelectTrigger>
              <SelectContent>
                {teams.map((team) => (
                  <SelectItem 
                    key={team} 
                    value={team}
                    disabled={team === teamA}
                  >
                    {team}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button
            onClick={handlePredict}
            disabled={!canPredict}
            className="min-w-32"
            size="lg"
          >
            {isLoading ? (
              <>
                <ReloadIcon className="mr-2 h-4 w-4 animate-spin" />
                Predicting...
              </>
            ) : (
              "Predict Match"
            )}
          </Button>
          
          <Button
            variant="outline"
            onClick={handleClear}
            className="min-w-32"
            size="lg"
          >
            Clear Inputs
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default TeamSelector;
