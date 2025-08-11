import { MoonIcon, SunIcon } from "@radix-ui/react-icons";
import { Button } from "./ui/Button";


function Header({ isDarkMode, toggleDarkMode }) {
  return (
    <header className="border-b bg-card shadow-sm">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
              <span className="text-primary-foreground">⚽</span>
            </div>
            <h1 className="text-2xl">Football Match Predictor</h1>
          </div>
          
          <Button
            variant="outline"
            size="sm"
            onClick={toggleDarkMode}
            className="w-9 h-9 p-0 invisible"
          >
            {isDarkMode ? (
              <SunIcon className="h-4 w-4" />
            ) : (
              <MoonIcon className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>
    </header>
  );
}

export default Header;