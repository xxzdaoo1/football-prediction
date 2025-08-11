import React from "react";

function Footer() {
    return (
      <footer className="border-t bg-card mt-auto">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0">
            <div className="flex items-center space-x-2">
              <span className="text-muted-foreground">© 2025 Football Match Predictor</span>
            </div>
            
            <nav className="flex items-center space-x-6">
              <a 
                href="#" 
                className="text-muted-foreground hover:text-primary transition-colors"
              >
                Home
              </a>
              <a 
                href="#" 
                className="text-muted-foreground hover:text-primary transition-colors"
              >
                About
              </a>
              <a 
                href="#" 
                className="text-muted-foreground hover:text-primary transition-colors"
              >
                History
              </a>
            </nav>
          </div>
        </div>
      </footer>
    );
  }

  export default Footer;