import { Card, CardContent, CardHeader, CardTitle } from "./ui/Card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/Table";
import { Badge } from "./ui/Badge";

const getStatusBadge = (status) => {
  switch (status) {
    case "correct":
      return <Badge variant="default" className="bg-secondary">Correct</Badge>;
    case "incorrect":
      return <Badge variant="destructive" >Incorrect</Badge>;
    default:
      return <Badge variant="secondary">Pending</Badge>;
  }
};

export default function HistorySection({ history }) {
  if (history.length === 0) {
    return (
      <Card className="w-full max-w-4xl mx-auto">
        <CardHeader>
          <CardTitle>Prediction History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <p>No predictions yet. Make your first prediction above!</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Prediction History</CardTitle>
        <p className="text-muted-foreground">
          Track your prediction accuracy over time
        </p>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Date</TableHead>
                <TableHead>Match</TableHead>
                <TableHead>Predicted</TableHead>
                <TableHead>Actual</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {history.map((item) => (
                <TableRow key={item.id}>
                  <TableCell className="whitespace-nowrap">
                    {item.date}
                  </TableCell>
                  <TableCell>
                    <div className="font-medium">
                      {item.teamA} vs {item.teamB}
                    </div>
                  </TableCell>
                  <TableCell>{item.predictedResult}</TableCell>
                  <TableCell>
                    {item.actualResult || "-"}
                  </TableCell>
                  <TableCell>
                    <span className="font-medium">{item.confidence}%</span>
                  </TableCell>
                  <TableCell>
                    {getStatusBadge(item.status)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}