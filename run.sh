#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}   Stock Market Analysis Launcher      ${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""

# Check if virtual environment exists, if not create it
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}Virtual environment created successfully.${NC}"
fi

# Activate the virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Check if dependencies are installed by attempting to import pandas
if ! python3 -c "import pandas, streamlit, torch" &> /dev/null; then
    echo -e "${YELLOW}Installing dependencies from requirements.txt...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed successfully.${NC}"
fi

echo -e "${GREEN}Environment is ready!${NC}"
echo ""
echo -e "Please select an option to run:"
echo -e "  ${YELLOW}1${NC}) Launch Interactive Streamlit Dashboard (app.py)"
echo -e "  ${YELLOW}2${NC}) Run CLI Pipeline with Sample Data (Choose Ticker)"
echo -e "  ${YELLOW}3${NC}) Run CLI Pipeline with Live Data (Choose Ticker)"
echo -e "  ${YELLOW}4${NC}) Run Unit Tests (pytest)"
echo -e "  ${YELLOW}q${NC}) Quit"
echo ""

read -p "Enter your choice [1-4, q]: " choice

case $choice in
    1)
        echo -e "\n${BLUE}Launching Streamlit Dashboard...${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop the server.${NC}"
        streamlit run app.py
        ;;
    2)
        read -p "Enter ticker symbol [default: NVDA]: " ticker
        ticker=${ticker:-NVDA}
        read -p "Target mode (price/percentage) [default: price]: " target_mode
        target_mode=${target_mode:-price}
        read -p "Save output to disk? (y/n) [default: n]: " save_output
        save_output=${save_output:-n}

        save_flag="--no-save"
        if [[ "$save_output" =~ ^[Yy]$ ]]; then
            save_flag=""
        fi

        echo -e "\n${BLUE}Running CLI Pipeline with Sample Data for $ticker (Predicting ${target_mode}, 25 Epochs)...${NC}"
        python run_pipeline.py --symbol "$ticker" --target "$target_mode" $save_flag --epochs 25
        ;;
    3)
        read -p "Enter ticker symbol [default: AAPL]: " ticker
        ticker=${ticker:-AAPL}
        read -p "Target mode (price/percentage) [default: price]: " target_mode
        target_mode=${target_mode:-price}

        read -p "Data interval (1d/1h/1wk) [default: 1d]: " interval
        interval=${interval:-1d}

        if [[ "$interval" == "1h" || "$interval" == "60m" ]]; then
            read -p "Historical data period (1y/2y - max 2y for 1h interval) [default: 2y]: " period
            period=${period:-2y}
            if [[ "$period" =~ ^(3y|4y|5y)$ ]]; then
                echo -e "${YELLOW}Note: 1h interval is restricted to max 2y by Yahoo Finance. Setting period to 2y.${NC}"
                period="2y"
            fi
        else
            read -p "Historical data period (1y/2y/3y/4y/5y) [default: 2y]: " period
            period=${period:-2y}
        fi

        read -p "Save output to disk? (y/n) [default: n]: " save_output
        save_output=${save_output:-n}

        save_flag="--no-save"
        if [[ "$save_output" =~ ^[Yy]$ ]]; then
            save_flag=""
        fi

        echo -e "\n${BLUE}Running CLI Pipeline with Live Data for $ticker (Target: ${target_mode}, Interval: ${interval}, Period: ${period}, 25 Epochs)...${NC}"
        python run_pipeline.py --symbol "$ticker" --target "$target_mode" --interval "$interval" --period "$period" --live $save_flag --epochs 25
        ;;
    4)
        echo -e "\n${BLUE}Running Unit Tests...${NC}"
        pytest tests/ -v
        ;;
    q|Q)
        echo -e "${GREEN}Exiting.${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        ;;
esac

# Deactivate when script ends
deactivate 2>/dev/null || true
