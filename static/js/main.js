/**
 * TMHNA Financial Intelligence Portal
 * Main JavaScript - Modern Enterprise UI v2.0
 */

document.addEventListener('DOMContentLoaded', function() {
    initFlashMessages();
    initSidebar();
    initSearch();
    initFormValidation();
    initTooltips();
    initAnimations();
});

/**
 * Auto-dismiss flash messages with animation
 */
function initFlashMessages() {
    const flashMessages = document.querySelectorAll('.flash-message');
    
    flashMessages.forEach(function(message) {
        // Add close button
        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = '×';
        closeBtn.style.cssText = `
            background: none;
            border: none;
            font-size: 1.25rem;
            color: inherit;
            opacity: 0.6;
            cursor: pointer;
            padding: 0;
            margin-left: auto;
            line-height: 1;
        `;
        closeBtn.addEventListener('click', () => dismissMessage(message));
        message.style.display = 'flex';
        message.style.alignItems = 'center';
        message.appendChild(closeBtn);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => dismissMessage(message), 5000);
    });
}

function dismissMessage(message) {
    message.style.transition = 'all 0.3s ease';
    message.style.opacity = '0';
    message.style.transform = 'translateY(-10px)';
    setTimeout(() => message.remove(), 300);
}

/**
 * Sidebar responsive toggle
 */
function initSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    
    // Create mobile menu toggle button
    if (window.innerWidth <= 768 && sidebar) {
        const menuToggle = document.createElement('button');
        menuToggle.className = 'topbar-btn mobile-menu-toggle';
        menuToggle.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="3" y1="12" x2="21" y2="12"></line>
                <line x1="3" y1="6" x2="21" y2="6"></line>
                <line x1="3" y1="18" x2="21" y2="18"></line>
            </svg>
        `;
        menuToggle.style.marginRight = 'auto';
        
        const topbarLeft = document.querySelector('.topbar-left');
        if (topbarLeft) {
            topbarLeft.insertBefore(menuToggle, topbarLeft.firstChild);
        }
        
        menuToggle.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });
        
        // Close sidebar when clicking outside
        document.addEventListener('click', (e) => {
            if (sidebar.classList.contains('open') && 
                !sidebar.contains(e.target) && 
                !menuToggle.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        });
    }
}

/**
 * Search functionality
 */
function initSearch() {
    const searchInput = document.querySelector('.search-input');
    
    if (searchInput) {
        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const query = searchInput.value.trim();
                if (query) {
                    // In a real app, this would search entries
                    console.log('Search query:', query);
                    showNotification('Search functionality coming soon!', 'info');
                }
            }
        });
        
        // Focus with keyboard shortcut
        document.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                searchInput.focus();
            }
        });
    }
}

/**
 * Form validation for feedback forms
 */
function initFormValidation() {
    const feedbackForms = document.querySelectorAll('.feedback-form');
    
    feedbackForms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const textarea = form.querySelector('textarea');
            
            if (textarea && textarea.value.trim() === '') {
                e.preventDefault();
                textarea.style.borderColor = 'var(--error-500)';
                textarea.style.boxShadow = '0 0 0 3px rgba(239, 68, 68, 0.1)';
                textarea.focus();
                
                showNotification('Please provide feedback before submitting.', 'warning');
                
                setTimeout(() => {
                    textarea.style.borderColor = '';
                    textarea.style.boxShadow = '';
                }, 3000);
            }
        });
    });
}

/**
 * Initialize tooltips
 */
function initTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        let tooltip = null;
        
        element.addEventListener('mouseenter', (e) => {
            tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = element.getAttribute('data-tooltip');
            tooltip.style.cssText = `
                position: fixed;
                background: var(--secondary-900);
                color: white;
                padding: 0.5rem 0.75rem;
                border-radius: var(--radius-md);
                font-size: var(--text-xs);
                z-index: 1000;
                max-width: 200px;
                box-shadow: var(--shadow-lg);
                pointer-events: none;
                opacity: 0;
                transform: translateY(4px);
                transition: all 0.2s ease;
            `;
            
            document.body.appendChild(tooltip);
            
            const rect = element.getBoundingClientRect();
            const tooltipRect = tooltip.getBoundingClientRect();
            
            tooltip.style.top = (rect.top - tooltipRect.height - 8) + 'px';
            tooltip.style.left = (rect.left + (rect.width / 2) - (tooltipRect.width / 2)) + 'px';
            
            requestAnimationFrame(() => {
                tooltip.style.opacity = '1';
                tooltip.style.transform = 'translateY(0)';
            });
        });
        
        element.addEventListener('mouseleave', () => {
            if (tooltip) {
                tooltip.style.opacity = '0';
                tooltip.style.transform = 'translateY(4px)';
                setTimeout(() => {
                    if (tooltip && tooltip.parentNode) {
                        tooltip.parentNode.removeChild(tooltip);
                    }
                }, 200);
            }
        });
    });
}

/**
 * Initialize page animations
 */
function initAnimations() {
    // Animate KPI cards on scroll
    const kpiCards = document.querySelectorAll('.kpi-card');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                entry.target.style.animation = `fadeInUp 0.4s ease ${index * 0.1}s forwards`;
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });
    
    kpiCards.forEach(card => {
        card.style.opacity = '0';
        observer.observe(card);
    });
    
    // Animate entry cards
    const entryCards = document.querySelectorAll('.entry-card');
    const entryObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                entry.target.style.animation = `fadeInUp 0.3s ease ${index * 0.05}s forwards`;
                entryObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.05 });
    
    entryCards.forEach(card => {
        card.style.opacity = '0';
        entryObserver.observe(card);
    });
}

/**
 * Show notification toast
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        background: ${type === 'success' ? 'var(--success-600)' : 
                     type === 'warning' ? 'var(--warning-600)' : 
                     type === 'error' ? 'var(--error-600)' : 'var(--secondary-800)'};
        color: white;
        padding: 12px 20px;
        border-radius: var(--radius-lg);
        font-size: var(--text-sm);
        font-weight: 500;
        z-index: 1000;
        box-shadow: var(--shadow-xl);
        display: flex;
        align-items: center;
        gap: 8px;
        opacity: 0;
        transform: translateY(20px);
        transition: all 0.3s ease;
    `;
    
    const icons = {
        success: '✓',
        warning: '⚠',
        error: '✕',
        info: 'ℹ'
    };
    
    notification.innerHTML = `<span>${icons[type] || icons.info}</span> ${message}`;
    document.body.appendChild(notification);
    
    requestAnimationFrame(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateY(0)';
    });
    
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(20px)';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

/**
 * Format currency values
 */
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

/**
 * Format percentage values
 */
function formatPercent(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
    }).format(value);
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!', 'success');
    }).catch(() => {
        showNotification('Failed to copy', 'error');
    });
}

// CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
`;
document.head.appendChild(style);

// Export for use in templates
window.TMHNA = {
    showNotification,
    formatCurrency,
    formatPercent,
    copyToClipboard
};
